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



class Vocab:
    """
    A simple vocabulary class that takes an alphabet of symbols,
    and assigns each symbol a unique integer id., e.g.
    
    > v = Vocab(['a','b','c','d'])
    > v('c')
    2
    > v('a')
    0
    > len(v)
    4
    
    """
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.index_map = {letter: index for (index, letter) in 
                          list(enumerate(alphabet))}
        
    def __len__(self):
        return len(self.alphabet)
        
    def __call__(self, letter):
        return self.index_map[letter]


    
class Tensorize:
    """
    An instance of Tensorize is a function that maps a piece of data
    (i.e. a dictionary) to an input and output tensor for consumption by
    a neural network.

    """
    
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
    """
    A four layer CNN that uses max pooling to reduce the dimensionality of
    the input from 1024 to 64, then trains a fully connected neural network
    on the final layer.
    
    """    
    def __init__(self, n_input_symbols, hidden_size, 
                 kernel_size, output_classes,
                 input_symbol_vocab):
        super(SimpleCNN, self).__init__()
        
        self.input_symbol_vocab = input_symbol_vocab
        
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
    
    def get_input_vocab(self):
        return self.input_symbol_vocab
    
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
        

class CnnClassifier:
    """
    Wraps a trained neural network to use for classification.
    
    """    
    def __init__(self, cnn, tensorize):
        self.net = cnn
        self.tensorize = tensorize
     
    def run(self, data_loader):
        for data in data_loader:
            inputs, _ = self.tensorize(data)
            yield self.net(inputs)  
    
    def evaluate(self, data_loader):
        correct = 0
        total = 0       
        for data in data_loader:
            inputs, labels = self.tensorize(data)
            outputs = self.net(inputs) 
            correct_inc, total_inc = CnnClassifier.accuracy(outputs, labels)
            correct += correct_inc
            total += total_inc
        return correct / total
    
    @staticmethod
    def accuracy(outputs, labels):
        correct = 0
        total = 0
        for output, label in zip(outputs, labels):
            total += 1
            if label == output.argmax():
                correct += 1
        return correct, total

def train_network(net, tensorize, 
                  train_loader, dev_loader, 
                  n_epochs, learning_rate):
    """
    Trains a neural network using the provided DataLoaders (for training
    and development data).
    
    tensorize is a function that maps a piece of data (i.e. a dictionary)
    to an input and output tensor for the neural network.
    
    """
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
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
        for i, data in enumerate(train_loader, 0):
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
        accuracy = CnnClassifier(net, tensorize).evaluate(dev_loader)
        if accuracy >= best_accuracy:
            best_net = deepcopy(net)        
        print("Dev accuracy = {:.2f}".format(accuracy))
               
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return best_net
 
    


def train_all_categories(dataset, hidden_size=18, kernel_size=7):
    """
    Trains a multi-way CNN classifier for all protein families in the
    dataset.
    
    e.g. train_all_categories(JsonDataset('data/var7.10.json'))
    
    """   
    def extract_amino_acid_alphabet(dataset):
        symbols = set()
        for sequence in dataset.select('seq'):
            letters = set(sequence)
            symbols = symbols | letters
        return sorted(list(symbols))

    train_sampler, val_sampler, _ = get_samplers(range(len(dataset)), 
                                                 .05, .25)
    categories = sorted(list(set(dataset.select('family'))))
    category_vocab = Vocab(categories)
    char_vocab = Vocab(extract_amino_acid_alphabet(dataset))
    CNN = cudaify(SimpleCNN(len(char_vocab.alphabet),
                            hidden_size, 
                            kernel_size, 
                            len(category_vocab),
                            char_vocab))
    train_loader =  DataLoader(dataset, batch_size=32,
                               sampler=train_sampler, num_workers=2)
    dev_loader =  DataLoader(dataset, batch_size=5,
                             sampler=val_sampler, num_workers=2)
    train_network(CNN, Tensorize(char_vocab, category_vocab, 1024), 
                  train_loader, dev_loader, n_epochs=12, learning_rate=0.001)
    print_patterns(CNN)
    return CNN



def print_patterns(CNN, cutoff=0.2):
    """
    An attempt to interpret the important patterns discovered
    by the CNN kernels.    
    
    """    
    char_vocab = CNN.get_input_vocab()
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
        
