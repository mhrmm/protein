import time
import torch
import torch.nn.functional as F
from torch import optim
from cog import JsonDataset, get_samplers, batched_loader



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
        category = torch.Tensor([self.category_vocab(c) 
                             for c in data['family']]).long()
        return words, category
        
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
    
    n_batches = len(train_loader)
    print_every = n_batches // 10    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    training_start_time = time.time()
    for epoch in range(n_epochs):
        running_loss = 0.0
        start_time = time.time()
        total_train_loss = 0        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = tensorize(data)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]
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
            total_val_loss += val_loss_size.data[0]
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        print("Accuracy = {:.2f}".format(correct/total))
        
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
 
    
    
HIDDEN_SIZE = 18
KERNEL_SIZE = 7
PADDING = KERNEL_SIZE//2
   
    
all_data = JsonDataset('data/var7.10.json')
#all_data = JsonDataset('cog10000.json')
train_sampler, val_sampler, test_sampler = get_samplers(all_data, .25, .25)
  
category_vocab = Vocab(extract_categories(all_data))
char_vocab = Vocab(extract_alphabet(all_data))
CNN = SimpleCNN(len(char_vocab.alphabet), HIDDEN_SIZE, KERNEL_SIZE, len(category_vocab))

train_loader = batched_loader(all_data, train_sampler, 32)
val_loader = batched_loader(all_data, val_sampler, 5)

train_network(CNN, Tensorize(char_vocab, category_vocab, 1024), 
              train_loader, val_loader, n_epochs=12, learning_rate=0.001)

def print_patterns(CNN, cutoff=0.2):
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
        

