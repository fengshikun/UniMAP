from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
import multiprocessing
from tqdm import tqdm

import multiprocessing
from tqdm import tqdm

import numpy as np
from rdkit import Chem
from descriptastorus.descriptors import rdDescriptors
from typing import Callable, Union


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

Molecule = Union[str, Chem.Mol]
import argparse


RDKIT_PROPS = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
               'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
               'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
               'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
               'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
               'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
               'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
               'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
               'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
               'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
               'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
               'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
               'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
               'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
               'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
               'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
               'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']


def rdkit_functional_group_label_features_generator(mol: Molecule) -> np.ndarray:
    """
    Generates functional group label for a molecule using RDKit.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdDescriptors.RDKit2D(RDKIT_PROPS)
    features = generator.process(smiles)[1:]
    features = np.array(features)
    features[features != 0] = 1
    return features

def rdkit_2d_features_normalized_generator(mol: Molecule) -> np.ndarray:
    """
    Generates RDKit 2D normalized features for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D normalized features.
    """
    try:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
    except:
        # import pdb;pdb.set_trace()
        features = np.zeros(200)
    return features

def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
    """
    Generates RDKit 2D features for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdDescriptors.RDKit2D()
    features = generator.process(smiles)[1:]

    return features




def process_fg(line):
    smiles = line.strip()
    res = rdkit_functional_group_label_features_generator(smiles)
    return res


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

import io
def lines():
    with io.open(data_file, 'r', encoding='utf8', newline='\n') as srcf:
        for line in srcf:
            yield line.strip()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='extract efcp fingerprints and function group labels')
    parser.add_argument("--smiles_file", type=str, default="pubchme_smiles_file.lst")
   
    args = parser.parse_args()
    

    data_file = args.smiles_file

    total = len(io.open(data_file, 'r', encoding='utf8', newline='\n').readlines())



    pool = multiprocessing.Pool(64)

    rdkit_res = np.zeros((total, 85))
    cnt = 0
    for res in tqdm(pool.imap(process_fg, lines(), chunksize=100000), total=total):
        rdkit_res[cnt] = res
        cnt += 1

    np.save("fg_labels.npy", rdkit_res)

    ecfp_feat_res = np.zeros((total, 2048))
    cnt = 0
    for res in tqdm(pool.imap(get_bitvec, lines(), chunksize=100), total=total):
        ecfp_feat_res[cnt] = res
        cnt += 1

    np.save("ecfp_regression.npy", ecfp_feat_res)





