import random
import torch
from sklearn.model_selection import train_test_split

from encoding import wordToTensor

def trainDevTestSplit(data, dev_size, test_size):
    devtrain, test = train_test_split(data, test_size=test_size)
    train, dev = train_test_split(devtrain, test_size=dev_size/(1-test_size))
    return train, dev, test

class DataSplitter:
    
    def __init__(self, instances, dev_size, test_size):
        self.train = dict()
        self.dev = dict()
        self.test = dict()
        for category in instances:
            train, dev, test = trainDevTestSplit(instances[category], 
                                                 dev_size, 
                                                 test_size)
            self.train[category] = train
            self.dev[category] = dev
            self.test[category] = test
        self.vocab = set()
        for category in instances:
            for instance in instances[category]:
                for ch in instance:
                    self.vocab.add(ch)
        self.categories = instances.keys()
            
    
    
    
            

def generateData(generator, num_positive, num_negative, dev_size, test_size):
    """
    generator = VariableLengthKthLast(4, 10, 3)    
    category_words, dev_category_words, test_category_words = generateData(generator)
    
    """
    positive_instances = set()
    negative_instances = set()
    while len(positive_instances) < num_positive:
        candidate = generator.createRandomPositive()
        positive_instances.add(candidate)
    while len(negative_instances) < num_negative:
        candidate = generator.createRandomNegative()
        negative_instances.add(candidate)
    positive_instances = list(positive_instances)
    negative_instances = list(negative_instances)
    ptrain, pdev, ptest = trainDevTestSplit(positive_instances, dev_size, test_size)
    ntrain, ndev, ntest = trainDevTestSplit(negative_instances, dev_size, test_size)
    category_words = {'positive': ptrain, 'negative': ntrain}
    dev_category_words = {'positive': pdev, 'negative': ndev}
    test_category_words = {'positive': ptest, 'negative': ntest}
    return category_words, dev_category_words, test_category_words

def categoryFromOutput(output, categories):
    top_n, top_i = output.data.topk(1) 
    category_i = top_i[0][0]
    return categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


class DataManager:
    
    def __init__(self, vocab, categories, train, dev, test):
        self.train, self.dev, self.test = train, dev, test
        self.vocab = dict()
        for (i, letter) in enumerate(list(vocab)):
            self.vocab[letter] = i
        self.categories = list(categories)
    
    

    def randomTrainingExample(self):
        """
        manager = DataManager(VariableLengthKthLast(4, 10, 3), 5, 5, 0.2, 0.2)
        for i in range(10):
            category, word, category_tensor, word_tensor = randomTrainingExample()
            print('category =', category, '/ word =', word)
        
        """
        category = randomChoice(self.categories)
        word = randomChoice(self.train[category])
        category_tensor = torch.LongTensor([self.categories.index(category)])
        word_tensor = wordToTensor(word, self.vocab)
        return category, word, category_tensor, word_tensor
    
    def randomTestExample(self):
        category = randomChoice(self.categories)
        word = randomChoice(self.test[category])
        category_tensor = torch.LongTensor([self.categories.index(category)])
        word_tensor = wordToTensor(word, self.vocab)
        return category, word, category_tensor, word_tensor
    
    def devIterator(self):
        for category in self.categories:
            for word in self.dev[category]:
                category_tensor = torch.LongTensor([self.categories.index(category)])
                word_tensor = wordToTensor(word, self.vocab)
                yield (category, word, category_tensor, word_tensor)
    
                
    def testIterator(self):
        for category in self.categories:
            for word in self.test[category]:
                category_tensor = torch.LongTensor([self.categories.index(category)])
                word_tensor = wordToTensor(word, self.vocab)
                yield (category, word, category_tensor, word_tensor)


