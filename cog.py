from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
import json


def json_from_generator(json_file, generator, num_to_generate):
    """
    e.g. 
    from generation import VariableLengthKth
    json_from_generator('./var9.100-1000.json', VariableLengthKth(100, 1000, 9), 1000)

    """
    data = []
    for i in range(num_to_generate // 2):
        seq = generator.createRandomPositive()
        data.append({'seq': seq, 'family': 'positive'})
        seq = generator.createRandomNegative()
        data.append({'seq': seq, 'family': 'negative'})        
    with open(json_file, 'w') as handle:
        json.dump(data, handle)
        
from generation import VariableLengthKth
json_from_generator('./var7.10.json', VariableLengthKth(10, 11, 7), 1000)
    
class JsonDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file) as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def families(data):
    families = set()
    for datum in data:
        families.add(datum['family'])
    return families

def discrim_subset(data, family):
    in_family = set()
    for (i, datum) in enumerate(data):
        if datum['family'] == family:
            in_family.add(i)
    out_of_family = set()
    while len(out_of_family) < len(in_family):
        sample = random.randint(0, len(data) - 1)
        if sample not in in_family:
            out_of_family.add(sample)
    result = []
    for i in in_family:
        result.append(data[i])
    for i in out_of_family:
        datum = data[i]
        datum['family'] = 'negative'
        result.append(datum)
    return result
    
def discrim_subset_ids(data, family):
    in_family = set()
    for (i, datum) in enumerate(data):
        if datum['family'] == family:
            in_family.add(i)
    out_of_family = set()
    while len(out_of_family) < len(in_family):
        sample = random.randint(0, len(data) - 1)
        if sample not in in_family:
            out_of_family.add(sample)
    return sorted(list(in_family)), sorted(list(out_of_family))

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
    
def split_data(dataset, dev_percent, test_percent):
    all_ids = range(len(dataset))
    dev_size = int(dev_percent * len(all_ids))
    test_size = int(test_percent * len(all_ids))
    train_ids = set(all_ids)
    dev_ids = random.sample(train_ids, dev_size)
    train_ids = train_ids - set(dev_ids)
    test_ids = random.sample(train_ids, test_size)
    train_ids = list(train_ids - set(test_ids))
    return train_ids, dev_ids, test_ids
  
    

def get_samplers(dataset, all_ids, dev_percent, test_percent):
    dev_size = int(dev_percent * len(all_ids))
    test_size = int(test_percent * len(all_ids))
    train_ids = set(all_ids)
    dev_ids = random.sample(train_ids, dev_size)
    train_ids = train_ids - set(dev_ids)
    test_ids = random.sample(train_ids, test_size)
    train_ids = list(train_ids - set(test_ids))
    train_sampler = SubsetRandomSampler(train_ids)
    dev_sampler = SubsetRandomSampler(dev_ids)
    test_sampler = SubsetRandomSampler(test_ids)
    return train_sampler, dev_sampler, test_sampler

# DataLoader takes in a dataset and a sampler for loading 
# (num_workers deals with system level memory) 
def batched_loader(dataset, sampler, batch_size):
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=2)
    
    


