import selfies as sf
from torch.utils.data import Dataset
import os
import pickle
import torch
import sys
sys.path.insert(0,'..')
from utils.mol import smiles2graph
from typing import List, Dict
from torch_geometric.data import Data, Batch
from iupac_token import IUPACTokenizer, SmilesIUPACTokenizer, SmilesTokenizer
from torch.nn import functional as F
# data_folder:
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

class GenSelfiesDataset(Dataset):
    def __init__(self, data_folder, tokenizer,  sim_file='zinc250k.smi', smi_self_file='train_self.smi', vocab_file='vocab_lst.pkl', pkl_file='dm_info.pkl', max_seq_length=128):
        super().__init__()
        self.data_folder = data_folder
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.sim_file = os.path.join(self.data_folder, sim_file)
        self.smi_self_file = os.path.join(self.data_folder, smi_self_file)
        self.vocab_file = os.path.join(self.data_folder, vocab_file)

        self.smile_lst = []
        self.selfies = []
        self.alphabet = set()
        with open(self.sim_file, 'r') as sr:
            for line in sr:
                self.smile_lst.append(line.strip())

        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, "rb") as fp:
                self.alphabet = pickle.load(fp)
            with open(self.smi_self_file, 'r') as fr:
                for line in fr:
                    self.selfies.append(line.strip())
        elif pkl_file is not None:
            pkl_file_path = os.path.join(self.data_folder, pkl_file)
            with open(pkl_file_path, "rb") as fp:
                dataset_info = pickle.load(fp)
                self.alphabet = dataset_info['alphabet']
                self.max_len = dataset_info['max_len'] + 1
                self.symbol_to_idx = dataset_info['symbol_to_idx']
                self.idx_to_symbol = dataset_info['idx_to_symbol']
                self.encodings = dataset_info['encodings']
                self.pad_idx = len(self.symbol_to_idx)
        else:
            self.generate_file()

        self.length = len(self.smile_lst)

        if pkl_file is None:
            self.max_len = max(len(list(sf.split_selfies(s))) for s in self.selfies)
            self.symbol_to_idx = {s: i for i, s in enumerate(self.alphabet)}
            self.idx_to_symbol = {i: s for i, s in enumerate(self.alphabet)}
            self.encodings = [[self.symbol_to_idx[symbol] for symbol in sf.split_selfies(s)] for s in self.selfies]



    def generate_file(self):
        assert os.path.exists(self.sim_file)
        
        for smi in self.smile_lst:
            self.selfies.append(sf.encoder(smi))
        for s in self.selfies:
            self.alphabet.update(sf.split_selfies(s))
        self.alphabet = ['[nop]'] + list(sorted(self.alphabet))

        # save self.selfies self.alphabet
        with open(self.vocab_file, "wb") as fp:
            pickle.dump(self.alphabet, fp)
        
        # save self.selfiles
        with open(self.smi_self_file, 'w') as fw:
            for line in self.selfies:
                fw.write(f"{line}\n")
        
        pass

    def __len__(self):
        return self.length
    
    def _getitem_smi(self, smi):
        item = {}
        inputs = self.tokenizer(
                smi,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
        )
        item['input_ids'] = inputs['input_ids']
        item['attention_mask'] = inputs['attention_mask']
        graph, _ = smiles2graph(smi)
        item['graph'] = graph
        return item
    
    def __getitem__(self, i):
        item = {}
        item['target_encoding'] = torch.tensor(self.encodings[i] + [self.symbol_to_idx['[nop]']] + [self.pad_idx for i in range(self.max_len - len(self.encodings[i]))])
        smi = self.smile_lst[i]
        inputs = self.tokenizer(
                smi,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
        )
        item['input_ids'] = inputs['input_ids']
        item['attention_mask'] = inputs['attention_mask']
        graph, _ = smiles2graph(smi)
        item['graph'] = graph
        return item
    
    def one_hot_to_selfies(self, hot):
        return ''.join([self.idx_to_symbol[idx.item()] for idx in hot.view((self.max_len, -1)).argmax(1)]).replace(' ', '')

    def one_hot_to_selfies_multi(self, hot):
        hot = hot.view(self.max_len, -1)
        hot = F.softmax(hot, dim=-1)
        hot_idx = [torch.multinomial(hot[i], num_samples=1)[0] for i in range(self.max_len)]
        return ''.join([self.idx_to_symbol[idx.item()] for idx in hot_idx]).replace(' ', '')

    def one_hot_to_smiles(self, hot):
        # return sf.decoder(self.one_hot_to_selfies(hot))
        return sf.decoder(self.one_hot_to_selfies_multi(hot))

def default_data_collator(features: List[Dict]) -> Dict[str, torch.Tensor]:
    first = features[0]
    batch = {}

    graph_lst = []
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, Data):
                graph_lst = [f[k] for f in features]
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    
    if len(graph_lst):
        batch['graph'] = Batch.from_data_list(graph_lst)
    
    
    return batch


class Dataset(Dataset):
    def __init__(self, file):
        selfies = [sf.encoder(line.split()[0]) for line in open(file, 'r')]
        self.alphabet = set()
        for s in selfies:
            self.alphabet.update(sf.split_selfies(s))
        self.alphabet = ['[nop]'] + list(sorted(self.alphabet))
        self.max_len = max(len(list(sf.split_selfies(s))) for s in selfies)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.alphabet)}
        self.idx_to_symbol = {i: s for i, s in enumerate(self.alphabet)}
        self.encodings = [[self.symbol_to_idx[symbol] for symbol in sf.split_selfies(s)] for s in selfies]
        
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, i):
        return torch.tensor(self.encodings[i] + [self.symbol_to_idx['[nop]'] for i in range(self.max_len - len(self.encodings[i]))])


class MolDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, file):
        super(MolDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset = Dataset(file)
        self.train_data, self.test_data = random_split(self.dataset, [int(round(len(self.dataset) * 0.8)), int(round(len(self.dataset) * 0.2))])
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=True, num_workers=16, pin_memory=True)

if __name__ == "__main__":
    smiles_tokenizer = SmilesTokenizer.from_pretrained('/home/fengshikun/iupac-pretrain_3/smiles_tokenizer', max_len=128)
    gen_self = GenSelfiesDataset(data_folder='/sharefs/sharefs-test_data/LIMO', tokenizer=smiles_tokenizer, sim_file='test.smi')