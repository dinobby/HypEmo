import torch
import numpy as np
import pandas as pd
from config import ENCODER_TYPE, all_dataset_list
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder

class HyoEmoDataSet(Dataset):
    def __init__(self, dataset, mode):
        super().__init__()
        assert dataset in all_dataset_list
        assert mode in ['train', 'valid', 'test']
        df = pd.read_csv(f'./data/{dataset}/{mode}.csv')
        self.text = df.text
        self.label = df.label
        self.tkr = AutoTokenizer.from_pretrained(ENCODER_TYPE)
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]
    
    def collate(self, batch):
        text = [t for t,_ in batch]
        encode = self.tkr(text, padding='longest', truncation= True, max_length=200, return_tensors='pt')
        label = [l for _,l in batch]
        label_tensor = torch.tensor(label)
        return encode, label_tensor


class HyoEmoDataSetForBert(Dataset):
    def __init__(self, dataset, mode, encoder_type='bert-base-uncased', label_included=None, upper_label=None):
        super().__init__()
        assert dataset in all_dataset_list
        assert mode in ['train', 'valid', 'test']
        assert not (label_included and upper_label)
        df = pd.read_csv(f'./data/{dataset}/{mode}.csv')
        le = LabelEncoder()
        if label_included:
            df = df.loc[df.label.isin(label_included)]
            df.reset_index(inplace=True)
            self.label = le.fit_transform(df.label)
        else:
            self.label = df.label
        if upper_label:
            self.label = [upper_label.get(i) for i in df.label]
        self.text = df.text
        self.tkr = AutoTokenizer.from_pretrained(encoder_type)
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]
            
    def collate(self, batch):
        text = [t for t,_ in batch]
        test_tensor = self.tkr(text, padding='longest', truncation= True, max_length=200, return_tensors='pt')
        label = [l for _,l in batch]
        label_tensor = torch.tensor(label)
        return test_tensor, label_tensor
