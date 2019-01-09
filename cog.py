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


def get_samplers(dataset, dev_percent, test_percent):
    dev_size = int(dev_percent * len(dataset))
    test_size = int(test_percent * len(dataset))
    train_ids = set(range(len(dataset)))
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
    
    


