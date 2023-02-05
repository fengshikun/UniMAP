
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
import numpy as np

# csv_file = '/sharefs/sharefs-test_data/PUBCHEM/iso_processed_txt/ALL_CSV/smiles.csv.can'
csv_file = '/home/AI4Science/fengsk/pubchem/iso_processed_txt/smiles.csv.can'


import multiprocess
from tqdm import tqdm
pool = multiprocess.Pool(32)

import io
def lines():
    with io.open(csv_file, 'r', encoding='utf8', newline='\n') as srcf:
        for line in srcf:
            yield line.strip()

total = len(io.open(csv_file, 'r', encoding='utf8', newline='\n').readlines())
# import pdb; pdb.set_trace()

def get_bitvec(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.full((2048), -1, dtype=np.int8)
        mol1 = Chem.AddHs(mol)
        fps1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048, useChirality=False)
        fp_array = np.zeros((0, ), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fps1, fp_array)
        return fp_array
    except:
        return np.full((2048), -1, dtype=np.int8)


ALL_FP = np.zeros((total, 2048), dtype=np.int8)

cnt = 0
for fp in tqdm(pool.imap(get_bitvec, lines(), chunksize=100), total=total):
    ALL_FP[cnt] = fp
    cnt += 1

# np.save('/sharefs/sharefs-test_data/PUBCHEM/iso_processed_txt/ALL_CSV/all_fp.npy', ALL_FP)
np.save('/home/AI4Science/fengsk/pubchem/iso_processed_txt/all_fp.npy', ALL_FP)
pool.close()
print('Finished')