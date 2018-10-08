
HUMAN_PROTEIN_FILENAME = '../Thesis/data/human.tab'

def read_proteins(filename):
    rows = []
    with open(filename) as inhandle:
        first_line = inhandle.readline().strip()
        headings = first_line.split('\t')
        for line in inhandle:
            fields = line.strip().split('\t')
            row = {heading: content for (heading, content) in zip(headings, fields)}
            rows.append(row)
    return rows



def group_by(proteins, column_heading):
    from collections import defaultdict 
    res = defaultdict(list)
    for protein in proteins:
        v = protein[column_heading]
        res[v].append(protein)
    return dict(res)


from wrangling import DataSplitter, DataManager

def get_protein_sequence_manager():
    proteins = read_proteins(HUMAN_PROTEIN_FILENAME)
    grouped = group_by(proteins, 'Protein families')
    family_counts = sorted([(len(grouped[k]), k) for k in grouped if k != ''])
    
    top_family = family_counts[-1][1]
    second_family = family_counts[-2][1]
    
    top_family_proteins = [p['Sequence'] for p in grouped[top_family]]
    second_family_proteins = [p['Sequence'] for p in grouped[second_family]]
    
    splitter = DataSplitter({'positive': top_family_proteins, 
                             'negative': second_family_proteins}, 
                            dev_size = 0.2, test_size = 0.2)
    print(splitter.vocab)
    
    
    return DataManager(splitter.vocab, splitter.categories, 
                       splitter.train, splitter.dev, splitter.test)

