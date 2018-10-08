import torch

def letterToIndex(letter, vocab):
    """
    Finds letter index from all_letters, e.g. "a" = 0
    
    """
    #return vocab.find(letter)
    return vocab[letter]

def letterToTensor(letter, vocab):
    """
    Just for demonstration, turn a letter into a <1 x n_letters> Tensor.
    
    """
    tensor = torch.zeros(1, len(vocab))
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def wordToTensor(word, vocab):
    """
    Turns a word into a <word_length x 1 x n_letters>,
    or an array of one-hot letter vectors.

    e.g.
        print(wordToTensor('BAD').size())
        print(wordToTensor('BAD').view(3, 7))   

    """

    tensor = torch.zeros(len(word), 1, len(vocab))
    for li, letter in enumerate(word):
        tensor[li][0][letterToIndex(letter, vocab)] = 1
    return tensor

