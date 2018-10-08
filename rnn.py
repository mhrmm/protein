import torch.nn as nn
import torch
from encoding import wordToTensor

class EncoderRNN(nn.Module):
    """
    n_hidden = 5
    all_letters = 'ABCDEFG'
    n_letters = len(all_letters)
    all_categories = ['positive', 'negative']
    n_categories = len(all_categories)
    rnn = EncoderRNN(n_letters, n_hidden, n_categories)
    
    input = wordToTensor('BAD', all_letters)
    hidden = torch.zeros(1, n_hidden)
    output, next_hidden = rnn(input[0], hidden)
    print(next_hidden)
    print(output)
    criterion = nn.NLLLoss()
    line_tensor = wordToTensor('CAGE', all_letters)
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        print('output: {}'.format(output))
        print('hidden: {}'.format(hidden))
        
    category_tensor = torch.tensor([ 0])
    loss = criterion(output.view(1, -1), category_tensor)
    print(loss)
    category_tensor = torch.tensor([ 1])
    loss = criterion(output.view(1, -1), category_tensor)
    print(loss)

    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        embedded = input.view(1, 1, -1)
        # embedded.shape = 1x1x7 (vocab size)
        hidden = hidden.view(1, 1, -1)
        # hidden.shape = 1x1x5 (hidden_size)
        output, hidden = self.gru(embedded, hidden)
        # output.shape = 1x1x5 (hidden_size)
        # hidden.shape = 1x1x5 (hidden_size)
        output = self.i2o(output)
        # output.shape = 1x1x2 (num categories)
        output = self.softmax(output)
        # output.shape = 1x1x2 (num categories)
        return output.view(1, -1), hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



