"""
This attempts to duplicate the data processing from
    Seo, Seokjun et al. 
    “DeepFam: deep learning based alignment-free method for protein family 
    modeling and prediction.” 
    Bioinformatics (2018).

They do a number of filtering operations on the COG (Clusters of 
Orthologous Groups) protein database from:
    
    https://www.ncbi.nlm.nih.gov/COG/

"""

from collections import defaultdict
import json

"""
Download these from:
  ftp://ftp.ncbi.nih.gov/pub/COG/COG2014/data

"""
COG_FILE = '../data/cog/cog2003-2014.csv'
DATA_FILE = '../data/cog/prot2003-2014.fa'
MAX_PROTEIN_LENGTH = 1000 # from Seo et al 2018

def get_protein_family_info(filename):
    family_sizes = defaultdict(int)
    protein_family = dict()
    with open(filename, 'r') as datahandle:
        for line in datahandle:
            fields = line.split(',')  
            family = fields[6].strip()
            family_sizes[family] += 1
            protein = fields[2].strip()
            protein_family[protein] = family
    return protein_family, family_sizes
    

def get_protein_sequences(filename):
    protein_id = None
    protein = ''
    proteins = {}
    with open(filename, 'r') as datahandle:
        for line in datahandle:
            if line.startswith('>gi'):
                if protein_id is not None:
                    proteins[protein_id] = protein
                    protein = ''
                fields = line.split('|')
                protein_id = fields[1].strip()
            elif protein_id is not None:
                protein += line.strip()
    proteins[protein_id] = protein
    return proteins

def get_subset(protein_family, family_sizes, proteins, minimum_family_size):
    unified = []
    total = 0
    short = 0
    quorum = 0
    for protein_id in proteins:
        if protein_id in protein_family:
            total += 1
            family = protein_family[protein_id]
            entry = {'id': int(protein_id), 
                     'family': family,
                     'seq': proteins[protein_id]}
            if len(proteins[protein_id]) <= MAX_PROTEIN_LENGTH:                
                short += 1
                if family_sizes[family] >= minimum_family_size:
                    quorum += 1
                    unified.append(entry)   
    print('Original number of proteins: {}'.format(total))
    print('After length filtering: {}'.format(short))
    print('After quorum filtering: {}'.format(quorum))
    return unified

protein_family, family_sizes = get_protein_family_info(COG_FILE)
proteins = get_protein_sequences(DATA_FILE)
unified = get_subset(protein_family, family_sizes, proteins, 10000)
with open('cog10000.json', 'w') as outfile:
    json.dump(unified, outfile)  

"""
protein_family, family_sizes = get_protein_family_info(COG_FILE)
proteins = get_protein_sequences(DATA_FILE)
unified = get_subset(protein_family, family_sizes, proteins, 100)
with open('cog100.json', 'w') as outfile:
    json.dump(unified, outfile)  
unified = get_subset(protein_family, family_sizes, proteins, 250)
with open('cog250.json', 'w') as outfile:
    json.dump(unified, outfile)
unified = get_subset(protein_family, family_sizes, proteins, 500)
with open('cog500.json', 'w') as outfile:
    json.dump(unified, outfile)
"""