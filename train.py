from rnn import EncoderRNN
from generation import VariableLengthKthLast
from wrangling import DataManager, categoryFromOutput, generateData
import time
import math
from torch import nn


def train(rnn, criterion, learning_rate, category_tensor, word_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        if p.grad is not None:
            p.data.add_(-learning_rate, p.grad.data)
    return output, loss.data[0]



# Go through a bunch of examples and record which are correctly guessed
def evaluateOnDev(rnn, manager, all_categories):
    def evaluate(rnn, word_tensor):
        hidden = rnn.initHidden()
        for i in range(word_tensor.size()[0]):
            output, hidden = rnn(word_tensor[i], hidden)
        return output
    correct = 0
    incorrect = 0
    for (category, word, category_tensor, word_tensor) in manager.devIterator():
        output = evaluate(rnn, word_tensor)
        guess, guess_i = categoryFromOutput(output, manager.categories)
        category_i = all_categories.index(category)
        if guess_i != category_i:
            incorrect += 1
        else:
            correct += 1
    return correct / float(correct + incorrect)

def runTraining(rnn, manager, n_iters, criterion, learning_rate, all_categories):
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    print_every = 100
    plot_every = 1000
    current_loss = 0
    dev_accuracy = 0
    all_losses = []
    all_dev_scores = []    
    start = time.time()
    for iter in range(1, n_iters + 1):
        category, word, category_tensor, word_tensor = manager.randomTrainingExample()
        output, loss = train(rnn, criterion, learning_rate, category_tensor, word_tensor)
        current_loss += loss

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            dev_accuracy = evaluateOnDev(rnn, manager, all_categories)
            all_dev_scores.append(dev_accuracy)
            current_loss = 0
            
        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, manager.categories)
            print("evaluating intermediate results...")
            dev_accuracy = evaluateOnDev(rnn, manager, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.3f %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), dev_accuracy, loss, word, guess, correct))

        
    return all_losses, all_dev_scores


def trainAndEvaluate(manager):
    n_letters = len(manager.vocab)
    n_hidden = 10
    criterion = nn.NLLLoss()
    learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
    
    loss_graphs = []
    dev_graphs = []
    
    rnn = EncoderRNN(n_letters, n_hidden, len(manager.categories))
    loss_graph, dev_graph = runTraining(rnn, manager, 20000, criterion, 
                                        learning_rate, manager.categories)
    
    loss_graphs.append(loss_graph)
    dev_graphs.append(dev_graph)    
    import matplotlib.pyplot as plt
    fig, (ax2, ax4) = plt.subplots(2, 1)
    fig.subplots_adjust(left=0.2, wspace=0.6)
    for i in range(len(loss_graphs)):
        ax2.plot(loss_graphs[i])
    ax2.set_ylabel('average loss')
    ax2.set_ylim(0, 1)    
    for i in range(len(dev_graphs)):
        ax4.plot(dev_graphs[i])
    ax4.set_ylabel('dev accuracy')
    ax4.set_ylim(0, 1)


def exampleTraining():
    
    generator = VariableLengthKthLast(4, 10, 3)
    train, dev, test = generateData(generator, 500, 500, 0.2, 0.2)

    manager = DataManager(generator.vocab, generator.getCategories(), train, dev, test)
    trainAndEvaluate(manager)


def proteinTraining():
    from data import get_protein_sequence_manager
    manager = get_protein_sequence_manager()
    #print(manager.randomTrainingExample())
    trainAndEvaluate(manager)
