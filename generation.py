import random


def generateRandomString(length, letters):
    characters = [random.choice(letters) for l in range(length)]
    return ''.join(characters)

def generateBuriedString(length, letters, buried):
    s = generateRandomString(length, letters)
    start_index = random.randint(0, len(s) - len(buried))
    return s[:start_index] + buried + s[start_index + len(buried):]
   
# Define the set of characters we will be creating an RNN for.
ALL_LETTERS = 'ABCDEFG'

class FixedLengthKthLast:
    def __init__(self, length, k):
        self.length = length
        self.k = k
        self.vocab = dict()
        for (i, letter) in enumerate(ALL_LETTERS):
            self.vocab[letter] = i
    
    def getCategories(self):
        return ['positive', 'negative']
    
    def isPositive(self, word):
        return word[-self.k] == 'G'

    def createRandomPositive(self):
        negative = self.createRandomNegative()
        return negative[:self.k] + 'G' + negative[-self.k+1:]

    def createRandomNegative(self):
        result = 'ZZGZ'
        while self.isPositive(result):
            result_length = self.length
            result = ''.join([random.choice(ALL_LETTERS) for i in range(result_length)])
        return result
    
class VariableLengthKthLast:
    """
    Randomly generates a word between LBOUND and UBOUND letters, such that the
    Kth last letter is 'g'.
    
    generator = VariableLengthKthLast(4, 10, 3)    
    for i in range(5):
        print('positive: {}'.format(generator.createRandomPositive()))
        print('negative: {}'.format(generator.createRandomNegative()))
    
    """
    
    def __init__(self, lower_bound, upper_bound, k):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.k = k
        self.vocab = dict()
        for (i, letter) in enumerate(ALL_LETTERS):
            self.vocab[letter] = i

    def getCategories(self):
        return ['positive', 'negative']
    
    def isPositive(self, word):
        return word[-self.k-1] == 'G'

    def createRandom(self):
        result_length = random.randint(self.lower_bound, self.upper_bound)
        return ''.join([random.choice(ALL_LETTERS) for i in range(result_length)])
        
    def createRandomPositive(self):
        negative = self.createRandom()
        chars = list(negative)
        chars[-self.k-1] = 'G' 
        return ''.join(chars)

    def createRandomNegative(self):
        result = self.createRandom()
        while self.isPositive(result):
            result = self.createRandom()
        return result
    
class VariableLengthKth:
    """
    Randomly generates a word between LBOUND and UBOUND letters, such that the
    Kth letter is 'g'.
    
    generator = VariableLengthKthLast(4, 10, 3)    
    for i in range(5):
        print('positive: {}'.format(generator.createRandomPositive()))
        print('negative: {}'.format(generator.createRandomNegative()))
    
    """
    
    def __init__(self, lbound, ubound, k):
        self.lower_bound = lbound
        self.upper_bound = ubound
        self.k = k
        self.vocab = dict()
        for (i, letter) in enumerate(ALL_LETTERS):
            self.vocab[letter] = i

    def getCategories(self):
        return ['positive', 'negative']
    
    def isPositive(self, word):
        return word[self.k] == 'G'

    def createRandom(self):
        result_length = random.randint(self.lower_bound, self.upper_bound)
        return ''.join([random.choice(ALL_LETTERS) for i in range(result_length)])
        
    def createRandomPositive(self):
        negative = self.createRandom()
        return negative[:self.k] + 'G' + negative[self.k+1:]

    def createRandomNegative(self):
        result = self.createRandom()
        while self.isPositive(result):
            result = self.createRandom()
        return result

