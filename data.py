from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import random
import json


class JsonDataset(Dataset):
    """
    Creates a Torch Dataset from a JSON file.
    
    We assume the JSON file is a list of dictionaries, where each
    dictionary corresponds to a single datum.
    
    """    
    def __init__(self, json_file):
        with open(json_file) as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def select(self, field):
        for datum in self:
            yield datum[field]

    

def domain(data, category):
    """
    Returns the set of existing values for a particular category in a 
    list of training data.
    
    """
    result = set()
    for datum in data:
        result.add(datum[category])
    return result

    
def discriminative_subset(data, category, value):
    """
    For a given category value, returns a balanced set of data that have
    that category value and that have a different category value.
    
    For instance, discriminative_subset(data, 'family', 'cog243') returns
    the subset of data for which datum['family'] = 'cog243', and an
    equivalent sized subset of data (randomly selected) for which
    datum['family'] != 'cog243'.
    
    """
    in_family = set()
    for (i, datum) in enumerate(data):
        if datum[category] == value:
            in_family.add(i)
    out_of_family = set()
    while len(out_of_family) < len(in_family):
        sample = random.randint(0, len(data) - 1)
        if sample not in in_family:
            out_of_family.add(sample)
    return sorted(list(in_family)), sorted(list(out_of_family))


def split_data(ids, dev_percent, test_percent):
    """
    Given a list of datum ids and dev/test percentages, returns a partition
    (train, dev, test) of the datum ids.
    
    """
    dev_size = int(dev_percent * len(ids))
    test_size = int(test_percent * len(ids))
    train_ids = set(ids)
    dev_ids = random.sample(train_ids, dev_size)
    train_ids = train_ids - set(dev_ids)
    test_ids = random.sample(train_ids, test_size)
    train_ids = list(train_ids - set(test_ids))
    return train_ids, dev_ids, test_ids

def get_samplers(all_ids, dev_percent, test_percent):
    """
    Given a list of datum ids and dev/test percentages, makes a
    train/dev/test split and returns samplers for the three subsets.
    
    """        
    train_ids, dev_ids, test_ids = split_data(all_ids, dev_percent, test_percent)
    train_sampler = SubsetRandomSampler(train_ids)
    dev_sampler = SubsetRandomSampler(dev_ids)
    test_sampler = SubsetRandomSampler(test_ids)
    return train_sampler, dev_sampler, test_sampler    
    
    
def json_from_generator(json_file, generator, num_to_generate):
    """
    e.g. 
    from generation import VariableLengthKth
    json_from_generator('./var9.100-1000.json', 
                        VariableLengthKth(100, 1000, 9), 
                        1000)

    """
    data = []
    for i in range(num_to_generate // 2):
        seq = generator.createRandomPositive()
        data.append({'seq': seq, 'family': 'positive'})
        seq = generator.createRandomNegative()
        data.append({'seq': seq, 'family': 'negative'})        
    with open(json_file, 'w') as handle:
        json.dump(data, handle)


