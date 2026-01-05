import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json
import os

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs

# [Fix] Robust path handling for ESPF
# Try to find ESPF in current dir, or up one level, or relative to this file
base_path = os.path.dirname(os.path.abspath(__file__))
espf_dir = os.path.join(base_path, 'ESPF')

if not os.path.exists(espf_dir):
    # Fallback to current working directory
    if os.path.exists('./ESPF'):
        espf_dir = './ESPF'
    elif os.path.exists('../ESPF'):
        espf_dir = '../ESPF'
    else:
        print("Warning: ESPF directory not found. Please ensure it exists.")
        espf_dir = './ESPF'  # Default fallback

vocab_path_p = os.path.join(espf_dir, 'protein_codes_uniprot.txt')
bpe_codes_protein = codecs.open(vocab_path_p)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv_p = pd.read_csv(os.path.join(espf_dir, 'subword_units_map_uniprot.csv'))

idx2word_p = sub_csv_p['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

vocab_path_d = os.path.join(espf_dir, 'drug_codes_chembl.txt')
bpe_codes_drug = codecs.open(vocab_path_d)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv_d = pd.read_csv(os.path.join(espf_dir, 'subword_units_map_chembl.csv'))

idx2word_d = sub_csv_d['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

max_d = 205
max_p = 545


def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    return i, np.asarray(input_mask)


def drug2emb_encoder(x):
    max_d = 50
    # max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti

        # Expose encoders for external use (needed for train.py)
        # We wrap the global functions into a simple class structure or just use them directly
        self.obj_enc_d = self.DrugEncoderWrapper()
        self.obj_enc_p = self.ProteinEncoderWrapper()

    class DrugEncoderWrapper:
        def transform(self, smiles_list):
            # Optimized batch transformation
            # Returns dictionary with 'index' and 'mask'
            index_list = []
            mask_list = []
            for smi in smiles_list:
                d_v, input_mask_d = drug2emb_encoder(smi)
                index_list.append(d_v)
                mask_list.append(input_mask_d)
            return {'index': np.array(index_list), 'mask': np.array(mask_list)}

    class ProteinEncoderWrapper:
        def transform(self, protein_list):
            index_list = []
            mask_list = []
            for prot in protein_list:
                p_v, input_mask_p = protein2emb_encoder(prot)
                index_list.append(p_v)
                mask_list.append(input_mask_p)
            return {'index': np.array(index_list), 'mask': np.array(mask_list)}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        # d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']

        # d_v = drug2single_vector(d)
        d_v, input_mask_d = drug2emb_encoder(d)
        p_v, input_mask_p = protein2emb_encoder(p)

        # print(d_v.shape)
        # print(input_mask_d.shape)
        # print(p_v.shape)
        # print(input_mask_p.shape)
        y = self.labels[index]
        return d_v, p_v, input_mask_d, input_mask_p, y