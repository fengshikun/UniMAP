from rdkit.Chem import MolFromSmiles, QED
from utils.sascorer import calculateScore
from rdkit.Chem.Crippen import MolLogP
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import selfies as sf
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def calculate_tanimoto(smiles, smiles2):
    mol = Chem.MolFromSmiles(smiles)
    mol1 = Chem.AddHs(mol)
    fps1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048, useChirality=False)
    
    mol2 = Chem.MolFromSmiles(smiles2)
    mol2 = Chem.AddHs(mol2)
    fps2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048, useChirality=False)
    fp_sim = DataStructs.TanimotoSimilarity(fps1, fps2)

    return fp_sim

def one_hot_to_smiles(prob, idx_to_symbol):
    return sf.decoder(one_hot_to_selfies(prob, idx_to_symbol))

def one_hot_to_selfies(prob, idx_to_symbol):
    # todo ignore the pad_idx: 108, may out of idx_to_symbol's range
    return ''.join([idx_to_symbol[idx.item()] for idx in prob.argmax(1)]).replace(' ', '')

def one_hots_to_penalized_logp(probs_lst, idx_to_symbol, return_smi=False):
    logps = []
    smi_lst = []
    for i, prob in enumerate(probs_lst):
        smile = one_hot_to_smiles(prob, idx_to_symbol)
        smi_lst.append(smile)
        mol = MolFromSmiles(smile)
        penalized_logp = MolLogP(mol) - calculateScore(mol)
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) > 6:
                penalized_logp -= 1
        logps.append(penalized_logp)
    if return_smi:
        return logps, smi_lst
    return logps

def smiles_to_penalized_logp(smi_lst):
    logps = []
    for smile in tqdm(smi_lst):
        mol = MolFromSmiles(smile)
        penalized_logp = MolLogP(mol) - calculateScore(mol)
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) > 6:
                penalized_logp -= 1
        logps.append(penalized_logp)
    return logps


def smiles_to_indices(smiles, symbol_to_idx):
    encoding = [symbol_to_idx[symbol] for symbol in sf.split_selfies(sf.encoder(smiles))]
    return torch.tensor(encoding + [symbol_to_idx['[nop]']])


def smiles_to_one_hot(smiles, symbol_to_idx):
    idx_smi = smiles_to_indices(smiles, symbol_to_idx)
    out = torch.zeros((idx_smi.size(0), len(symbol_to_idx)))
    for i, index in enumerate(idx_smi):
        out[i][index] = 1
    return out.flatten()


def generate_training_mols(num_mols, prop_func, device, generator):
    with torch.no_grad():
        z = torch.randn((num_mols, 1024), device=device)
        x = generator(z)
        y = torch.tensor(prop_func(x), device=device).unsqueeze(1).float()
    return x, y

class PropertyPredictor(pl.LightningModule):
    def __init__(self, in_dim, learning_rate=0.001):
        super(PropertyPredictor, self).__init__()
        self.learning_rate = learning_rate
        self.fc = nn.Sequential(nn.Linear(in_dim, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1))
        
    def forward(self, x):
        return self.fc(x)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def loss_function(self, pred, real):
        return F.mse_loss(pred, real)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        self.log('val_loss', loss)
        return loss

class PropDataModule(pl.LightningDataModule):
    def __init__(self, x, y, batch_size):
        super(PropDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset = TensorDataset(x, y)
        self.train_data, self.test_data = random_split(self.dataset, [int(round(len(self.dataset) * 0.9)), int(round(len(self.dataset) * 0.1))])
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=True)
    