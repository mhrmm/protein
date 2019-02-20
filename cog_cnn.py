import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from data import JsonDataset, get_samplers
from data import discriminative_subset
from data import domain, split_data
from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy

if torch.cuda.is_available():
    print("using gpu")
    cuda = torch.device('cuda:0')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        return model.cuda()
else: 
    print("using cpu")
    cuda = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        return model


def extract_alphabet(dataset):
    symbols = set()
    for datum in dataset:
        sequence = datum['seq']
        symbols = symbols | set(sequence)
    return sorted(list(symbols))

def extract_categories(dataset):
    symbols = set()
    for datum in dataset:
        symbols.add(datum['family'])
    return sorted(list(symbols))
        

# DataLoader takes in a dataset and a sampler for loading 
# (num_workers deals with system level memory) 
def batched_loader(dataset, sampler, batch_size):
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=2)

class Vocab:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.index_map = {letter: index for (index, letter) in 
                          list(enumerate(alphabet))}
        
    def __len__(self):
        return len(self.alphabet)
        
    def __call__(self, letter):
        return self.index_map[letter]


    
class Tensorize:
    def __init__(self, symbol_vocab, category_vocab, max_word_length):
        self.symbol_vocab = symbol_vocab
        self.category_vocab = category_vocab
        self.max_word_length = max_word_length
    
    def __call__(self, data):
        words = Tensorize.words_to_tensor(data['seq'], 
                                              self.symbol_vocab, 
                                              self.max_word_length).float()
        category = LongTensor([self.category_vocab(c) 
                             for c in data['family']])
        return cudaify(words), cudaify(category)
        
    @staticmethod
    def words_to_tensor(words, vocab, max_word_length):
        """
        Turns an K-length list of words into a <K, len(vocab), max_word_length>
        tensor.
    
        e.g.
            t = words_to_tensor(['BAD', 'GAB'], Vocab('ABCDEFG'), 3)
            # t[0] is a matrix representations of 'BAD', where the jth
            # column is a one-hot vector for the jth letter
            print(t[0])
    
        """
        tensor = torch.zeros(len(words), len(vocab), max_word_length)
        for i, word in enumerate(words):
            for li, letter in enumerate(word):
                tensor[i][vocab(letter)][li] = 1
        return tensor


   

class SimpleCNN(torch.nn.Module):
    
    
    def __init__(self, n_input_symbols, hidden_size, kernel_size, output_classes):
        super(SimpleCNN, self).__init__()
        
        self.hidden_size = hidden_size
        padding = kernel_size // 2

        #Size changes from (n_input_symbols, 1024) to (HIDDEN_SIZE, 512)
        self.conv1 = torch.nn.Conv1d(n_input_symbols, hidden_size, 
                                     kernel_size=kernel_size, 
                                     stride=1, padding=padding)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        #Size changes from (HIDDEN_SIZE, 512) to (HIDDEN_SIZE, 256)
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, 
                                     kernel_size=kernel_size, 
                                     stride=1, padding=padding)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        #Size changes from (HIDDEN_SIZE, 256) to (HIDDEN_SIZE, 128)
        self.conv3 = torch.nn.Conv1d(hidden_size, hidden_size, 
                                     kernel_size=kernel_size, 
                                     stride=1, padding=padding)
        self.pool3 = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        #Size changes from (HIDDEN_SIZE, 128) to (HIDDEN_SIZE, 64)
        self.conv4 = torch.nn.Conv1d(hidden_size, hidden_size, 
                                     kernel_size=kernel_size, 
                                     stride=1, padding=padding)
        self.pool4 = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = torch.nn.Linear(hidden_size * 64, 64)
        self.fc2 = torch.nn.Linear(64, output_classes)
        
    def forward(self, x):      
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1, self.hidden_size * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        



def accuracy(outputs, labels):
    correct = 0
    total = 0
    for output, label in zip(outputs, labels):
        total += 1
        if label == output.argmax():
            correct += 1
    return correct, total

def train_network(net, tensorize, train_loader, val_loader, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #max_batches_per_epoch = 100
    #n_batches = min(max_batches_per_epoch, len(train_loader))
    n_batches = len(train_loader)
    print_every = n_batches // 10    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    training_start_time = time.time()
    best_accuracy = -1.0
    best_net = None
    for epoch in range(n_epochs):
        running_loss = 0.0
        start_time = time.time()
        total_train_loss = 0 
        #train_iter = iter(train_loader)
        #for i in range(max_batches_per_epoch):
        for i, data in enumerate(train_loader, 0):
            #data = next(train_iter)
            inputs, labels = tensorize(data)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), 
                        running_loss / print_every, time.time() - start_time))
                running_loss = 0.0
                start_time = time.time()
            
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        correct = 0
        total = 0
        for data in val_loader:
            inputs, labels = tensorize(data)
            val_outputs = net(inputs)            
            val_loss_size = loss(val_outputs, labels)
            correct_inc, total_inc = accuracy(val_outputs, labels)
            correct += correct_inc
            total += total_inc
            total_val_loss += val_loss_size.data.item()
        if correct/total >= best_accuracy:
            best_net = deepcopy(net)
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        print("Accuracy = {:.2f}".format(correct/total))
        
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return best_net
    
def train_all_categories():
    HIDDEN_SIZE = 18
    KERNEL_SIZE = 7
       
    #all_data = JsonDataset('data/var7.10.json')
    all_data = JsonDataset('cog10000.json')
    train_sampler, val_sampler, test_sampler = get_samplers(all_data, range(len(all_data)), .05, .25)
      
    category_vocab = Vocab(extract_categories(all_data))
    char_vocab = Vocab(extract_alphabet(all_data))
    CNN = SimpleCNN(len(char_vocab.alphabet), HIDDEN_SIZE, KERNEL_SIZE, len(category_vocab))
    CNN = cudaify(CNN)
    
    train_loader = batched_loader(all_data, train_sampler, 32)
    val_loader = batched_loader(all_data, val_sampler, 5)
    
    train_network(CNN, Tensorize(char_vocab, category_vocab, 1024), 
                  train_loader, val_loader, n_epochs=12, learning_rate=0.001)



class FamilyMembershipDataset(Dataset):
    def __init__(self, base_data, family):
        self.base_data = base_data
        self.family = family
            
    def __len__(self):
        return len(self.base_data)

    def __getitem__(self, idx):
        result = {k: v for (k,v) in self.base_data[idx].items()}        
        if result['family'] != self.family:
            result['family'] = 'negative'
        return result        
    
  



def train_one_vs_rest():
    HIDDEN_SIZE = 18
    KERNEL_SIZE = 7
          
    #all_data = JsonDataset('data/var7.10.json')
    #all_data = JsonDataset('cog10000.json')
    
    full_data = JsonDataset('cog500.json')
    all_families = list(domain(full_data, 'family'))
    train_ids, dev_ids, test_ids = split_data(range(len(full_data)), .05, .02)
    
    test_set_probs = {}
    for test_id in test_ids:
        test_set_probs[full_data[test_id]['id']] = []
    
    trained_families = set()
    for family in all_families[:5]:
        family_ids, non_family_ids = discriminative_subset(full_data, 'family', family)
        all_ids = family_ids + non_family_ids
        print('Dataset for {} has {} instances.'.format(family, len(all_ids)))
        
        all_data = FamilyMembershipDataset(full_data, family)    
        train_sampler = SubsetRandomSampler(list(set(train_ids) & set(all_ids)))
        val_sampler = SubsetRandomSampler(list(set(dev_ids) & set(all_ids)))
        test_sampler = SubsetRandomSampler(list(set(test_ids) & set(all_ids)))
    
        category_vocab = Vocab(extract_categories(all_data))
        char_vocab = Vocab(extract_alphabet(all_data))
        CNN = SimpleCNN(len(char_vocab.alphabet), HIDDEN_SIZE, KERNEL_SIZE, len(category_vocab))
        CNN = cudaify(CNN)
        
        train_loader = batched_loader(all_data, train_sampler, 32)
        val_loader = batched_loader(all_data, val_sampler, 5)
        test_loader = batched_loader(all_data, test_sampler, 5)
        tensorize = Tensorize(char_vocab, category_vocab, 1024)
        
        net = train_network(CNN, tensorize, 
                      train_loader, val_loader, n_epochs=12, learning_rate=0.001)
        
        #At the end of training, do a pass on the test set
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels = tensorize(data)
            outputs = net(inputs)            
            correct_inc, total_inc = accuracy(outputs, labels)
            correct += correct_inc
            total += total_inc
        if total > 0:
            print("Test set accuracy = {:.2f}".format(correct/total))
        
        # Record the probabilities for each test set protein
        full_test_sampler = SubsetRandomSampler(test_ids)
        sample_size = 1000
        full_test_loader = batched_loader(all_data, full_test_sampler, sample_size)
        for data in full_test_loader:
            inputs, labels = tensorize(data)
            outputs = net(inputs)
            #print(outputs.shape)
            for i in range(outputs.shape[0]):
                log_likelihood = outputs[i][category_vocab.index_map[family]].item()
                #print(log_likelihood)
                test_set_probs[data['id'][i].item()].append((log_likelihood, family))
                #print(data['id'][i])
        
        print("Done recording.")
        
        trained_families.add(family)
        correct = 0
        total = 0
        for i in test_ids:
            datum = full_data[i]
            if datum['family'] in trained_families:
                (_, likeliest) = max(test_set_probs[datum['id']])
                if likeliest == datum['family']:
                    correct += 1
                total += 1
        print("Trained {} out of {} families.".format(len(trained_families), len(all_families)))
        print("There are {} instances so far.".format(total))
        print("Current accuracy: {:.2f}".format(correct/total))


def print_patterns(CNN, char_vocab, cutoff=0.2):
    w = CNN.conv1.weight
    (num_channels, _, window_size) = w.shape
    for channel in range(num_channels):
        matrix = w[channel]
        strongest = ''
        for window_index in range(window_size):
            abs_max_index = matrix[:,window_index].abs().argmax().item()
            abs_max = matrix[:,window_index].abs().max().item()
            if abs_max > cutoff:
                strongest += char_vocab.alphabet[abs_max_index]
            else:
                strongest += "_"
        print(strongest)
        

train_all_categories()
