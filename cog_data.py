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

"""
Download these from:
  ftp://ftp.ncbi.nih.gov/pub/COG/COG2014/data

"""
COG_FILE = '../data/cog/cog2003-2014.csv'
DATA_FILE = '../data/cog/prot2003-2014.fa'
MAX_PROTEIN_LENGTH = 1000 # from Seo et al 2018

class COGDatabase:
    """
    Reads and stores the information present in a COG database.
    
    You may extract subsets of the database in a convenient JSON format
    by modifying the following code snippet (which takes the subset of
    proteins belonging to families of size 10000 or more).
        
        import json
        db = COGDatabase(COG_FILE)
        unified = db.get_subset(10000)
        with open('cog.10000.json', 'w') as outfile:
            json.dump(unified, outfile)  
    
    """
    
    def __init__(self, filename):
        self.protein_family, self.family_sizes = COGDatabase.get_protein_family_info(filename)
        self.proteins = COGDatabase.get_protein_sequences(DATA_FILE)

    def get_subset(self, minimum_family_size, 
                   max_protein_length = MAX_PROTEIN_LENGTH):
                
        """
        Creates a list of dictionaries, one dictionary for each protein
        in the database. An example of a dictionary:
            
            {'id': 103486759,
             'family': 'COG2204',
             'seq': 'MALDILIVD...'
            }
            
        where 'id' is the protein identifier, 'family' is the COG family
        to which the protein belongs, and 'seq' is the amino acid sequence
        of that protein.
        
        You should specify a minimum family size to filter rare families
        from the list. You may also specify a maximum protein length
        to filter overly long amino acid sequences from the list.
        
        """
        unified = []
        total = 0
        short = 0
        quorum = 0
        for protein_id in self.proteins:
            if protein_id in self.protein_family:
                total += 1
                family = self.protein_family[protein_id]
                entry = {'id': int(protein_id), 
                         'family': family,
                         'seq': self.proteins[protein_id]}
                if len(self.proteins[protein_id]) <= max_protein_length:                
                    short += 1
                    if self.family_sizes[family] >= minimum_family_size:
                        quorum += 1
                        unified.append(entry)   
        print('Original number of proteins: {}'.format(total))
        print('After length filtering: {}'.format(short))
        print('After quorum filtering: {}'.format(quorum))
        return unified
        
    
    @staticmethod
    def get_protein_family_info(filename):
        """
        From a COG data file, initializes two data structures:
            
        - protein_family is a dictionary that maps protein ids to their
          COG protein family (e.g. '404215315': 'COG0002')
        - family_sizes is a dictionary that maps COG protein families to the
          number of protein instances in that family (e.g. 'COG1140': 171 would
          indicate that the database includes 171 COG1140 proteins)
          
        """
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
    
    @staticmethod
    def get_protein_sequences(filename):
        """
        From a COG data file, initializes a dictionary that maps each protein
        ids to its corresponding amino acid sequence.
        
        e.g. '103486747': 'MKVRILGCGTSSGVPRIGNDWG...'
        
        """
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



