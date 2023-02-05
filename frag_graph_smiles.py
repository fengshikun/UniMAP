from collections import defaultdict
from multiprocessing import set_start_method

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import BRICS
from rdkit.Contrib.IFG.ifg import identify_functional_groups


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_clique_mol(mol, atoms):
    # get the fragment of clique
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    return new_mol



def canonicalize(smi, clear_stereo=False):
    mol = Chem.MolFromSmiles(smi)
    if clear_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def fragment_graph_cutbonds(mol, cut_bonds_set):
    mol_num = len(list(mol.GetAtoms()))
    bond_set = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bond_set.append([a1, a2])
    
    left_bond_set = []
    
    for ele in bond_set:
        if [ele[0], ele[1]] not in cut_bonds_set and \
            [ele[1], ele[0]] not in cut_bonds_set:
                left_bond_set.append(ele)            
    
    left_bond_set = [list(ele) for ele in list(left_bond_set)]
    graph = defaultdict(list)

    for x, y in left_bond_set:
        graph[x].append(y)
        graph[y].append(x)
    
    visited = set()

    labels = [-1 for _ in range(mol_num)]
    
    def dfs(i, lb=-1):
        visited.add(i)
        labels[i] = lb
        for j in graph[i]:
            if j not in visited:
                dfs(j, lb)
    
    lb = 0
    for i in range(mol_num):
        if i not in visited:
            dfs(i, lb)
            lb += 1
    
    return labels


def fragmeng_graph(smiles, addHs=False):
    # c_smiles = canonicalize(smiles)
    # collect bond
    mol = Chem.MolFromSmiles(smiles)
    mol_num = len(list(mol.GetAtoms()))
    bond_set = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bond_set.append([a1, a2])
    
    split_bonds = list(BRICS.FindBRICSBonds(mol))
    cut_bonds_set = [list(ele[0]) for ele in split_bonds]
    left_bond_set = []
    
    for ele in bond_set:
        if [ele[0], ele[1]] not in cut_bonds_set and \
            [ele[1], ele[0]] not in cut_bonds_set:
                left_bond_set.append(ele)            
    
    left_bond_set = [list(ele) for ele in list(left_bond_set)]
    graph = defaultdict(list)

    for x, y in left_bond_set:
        graph[x].append(y)
        graph[y].append(x)
    
    visited = set()

    labels = [-1 for _ in range(mol_num)]
    
    def dfs(i, lb=-1):
        visited.add(i)
        labels[i] = lb
        for j in graph[i]:
            if j not in visited:
                dfs(j, lb)
    
    lb = 0
    for i in range(mol_num):
        if i not in visited:
            dfs(i, lb)
            lb += 1
    
    if addHs:
        new_mo_lst = []
        new_mo = Chem.AddHs(mol)
        for atom in new_mo.GetAtoms():
            new_mo_lst.append(atom.GetSymbol())
        
        h_start_idx = len(labels)
        h_atoms_len = len(new_mo_lst) - h_start_idx
        h_labels = [-1 for _ in range(h_atoms_len)]

        H_bonds_lst = []
        for bond in new_mo.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if a1 >= h_start_idx:
                assert a2 < h_start_idx
            if a2 >= h_start_idx:
                assert a1 < h_start_idx
            if a1 >= h_start_idx or a2 >= h_start_idx:
                H_bonds_lst.append([a1, a2])
                if a1 >= h_start_idx:
                    h_labels[a1 - h_start_idx] = labels[a2]
                else:
                    h_labels[a2 - h_start_idx] = labels[a1]
        assert -1 not in h_labels # all filled

        return labels, h_labels
    
    return labels


def fragmeng_graph_dfs_iter(smiles):
    # c_smiles = canonicalize(smiles)
    # collect bond
    mol = Chem.MolFromSmiles(smiles)
    mol_num = len(list(mol.GetAtoms()))
    bond_set = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bond_set.append([a1, a2])
    
    split_bonds = list(BRICS.FindBRICSBonds(mol))
    cut_bonds_set = [list(ele[0]) for ele in split_bonds]
    left_bond_set = []
    
    for ele in bond_set:
        if [ele[0], ele[1]] not in cut_bonds_set and \
            [ele[1], ele[0]] not in cut_bonds_set:
                left_bond_set.append(ele)            
    
    left_bond_set = [list(ele) for ele in list(left_bond_set)]
    graph = defaultdict(list)

    for x, y in left_bond_set:
        graph[x].append(y)
        graph[y].append(x)
    
    visited = set()

    labels = [-1 for _ in range(mol_num)]
    
    def dfs(i, lb=-1):
        visited.add(i)
        labels[i] = lb
        for j in graph[i]:
            if j not in visited:
                dfs(j, lb)


    def iter_dfs(s, lb=-1):
        S, Q = set(), []
        Q.append(s)
        
        while Q:
            u = Q.pop()
            if u in S: continue
            S.add(u)
            Q.extend(graph[u])
            visited.add(u)
            labels[u] = lb
            # yield u
    
    lb = 0
    for i in range(mol_num):
        if i not in visited:
            iter_dfs(i, lb)
            lb += 1
    
    return labels



# tokenizer:
# smiles = 'CC(=O)NCCC1=CNc2c1cc(OC)cc2CC(=O)NCCc1c[nH]c2ccc(OC)cc12'
# p_smiles = canonicalize(smiles)
# print(p_smiles)


# atom_array = ['B', 'C',  'N',  'O', 'P',  'S', 'F', 'Cl', 'Br', 'b', 
#               'c', 'n', 'o', 'p', 's', 'f', 'cl', 'br']
# 
atom_dict = {
    # 'H': 1,
      '[H+]': 1,
      'He': 2,
      'Li': 3,
      'Be': 4,
      'B': 5,
      'b': 5,
      'C': 6,
      'c': 6,
      'N': 7,
      'n': 7,
      'O': 8,
      'o': 8,
      'F': 9,
      'Ne': 10,
      'Na': 11,
      'Mg': 12,
      'Al': 13,
      'Si': 14,
      'P': 15,
      'p': 15,
      'S': 16,
      's': 16,
      'Cl': 17,
      'Ar': 18,
      'K': 19,
      'Ca': 20,
    #   'Sc': 21,
      'Ti': 22,
      'V': 23,
      'Cr': 24,
      'Mn': 25,
      'Fe': 26,
      'Co': 27,
      'Ni': 28,
      'Cu': 29,
      'Zn': 30,
      'Ga': 31,
      'Ge': 32,
      'As': 33,
      'Se': 34,
      'Br': 35,
      'Kr': 36,
      'Rb': 37,
      'Sr': 38,
      'Y': 39,
      'Zr': 40,
      'Nb': 41,
      'Mo': 42,
      'Tc': 43,
      'Ru': 44,
      'Rh': 45,
      'Pd': 46,
      'Ag': 47,
      'Cd': 48,
    #   'In': 49,
    #   'Sn': 50,
      'Sb': 51,
      'Te': 52,
      'I': 53,
      'Xe': 54,
      'Cs': 55,
      'Ba': 56,
      'La': 57,
      'Ce': 58,
      'Pr': 59,
      'Nd': 60,
      'Pm': 61,
      'Sm': 62,
      'Eu': 63,
      'Gd': 64,
      'Tb': 65,
      'Dy': 66,
      'Ho': 67,
      'Er': 68,
      'Tm': 69,
      'Yb': 70,
      'Lu': 71,
      'Hf': 72,
      'Ta': 73,
      'W': 74,
      'Re': 75,
      'Os': 76,
      'Ir': 77,
      'Pt': 78,
      'Au': 79,
      'Hg': 80,
      'Tl': 81,
      'Pb': 82,
      'Bi': 83,
      'Po': 84,
      'At': 85,
      'Rn': 86,
      'Fr': 87,
      'Ra': 88,
      'Ac': 89,
      'Th': 90,
      'Pa': 91,
      'U': 92,
    #   'Np': 93,
      'Pu': 94,
      'Am': 95,
      'Cm': 96,
      'Bk': 97,
      'Cf': 98,
      'Es': 99,
      'Fm': 100,
      'Md': 101,
      'No': 102,
      'Lr': 103,
      'Rf': 104,
      'Db': 105,
      'Sg': 106,
      'Bh': 107,
      'Hs': 108,
      'Mt': 109,
      'Ds': 110,
      'Rg': 111,
    #   'Cn': 112,
      'Uut': 113,
      'Uuq': 114,
      'Uup': 115,
      'Uuh': 116,
      'Uus': 117,
      'Uuo': 118

}


# c_smiles = 'C1=C(C(C=C(C1O)Cl)Cl)Cl'
multi_catom_lst = []
single_atom_lst = []
for atom in atom_dict:
    if len(atom) == 2:
        multi_catom_lst.append(atom)
    if len(atom) == 1:
        single_atom_lst.append(atom)
    
multi_catom_lst.append('\[\d*H\+\]')
multi_catom_lst.append('\[H\-\]')
multi_catom_lst.append('\[HH\]')
multi_catom_lst.append('\[Sc\]')
multi_catom_lst.append('\[Sc\+3\]')
multi_catom_lst.append('\[\d+Sc\+3\]')
multi_catom_lst.append('\[ScH3\]')
multi_catom_lst.append('\[\d+Sc]')
multi_catom_lst.append('\[In\]')
multi_catom_lst.append('\[InH\]')
multi_catom_lst.append('\[InH2\]')
multi_catom_lst.append('\[InH3\]')
multi_catom_lst.append('\[In2\]')
multi_catom_lst.append('\[In\-\]')
multi_catom_lst.append('\[In\+\]')
multi_catom_lst.append('\[\d*In\+3\]')
multi_catom_lst.append('\[\d+In]')
multi_catom_lst.append('\[Sn\]')
multi_catom_lst.append('\[Sn\+\]')
multi_catom_lst.append('\[Sn\-\]')
multi_catom_lst.append('\[Sn\+2\]')
multi_catom_lst.append('\[Sn\+3\]')
multi_catom_lst.append('\[Sn\+4\]')
multi_catom_lst.append('\[\d+Sn]')
multi_catom_lst.append('\[SnH\]')
multi_catom_lst.append('\[SnH2\]')
multi_catom_lst.append('\[SnH2\+2\]')
multi_catom_lst.append('\[SnH3\]')
multi_catom_lst.append('\[SnH3\+\]')
multi_catom_lst.append('\[SnH4\]')
multi_catom_lst.append('\[117Sn\+\d+\]')
multi_catom_lst.append('\[\d*Np\]')
multi_catom_lst.append('\[Si\]')

multi_catom_lst.append('\[S\-]')
multi_catom_lst.append('\[n\-]')
multi_catom_lst.append('\[N\+]')
multi_catom_lst.append('\[H\]')
multi_catom_lst.append('\[1H\]')
multi_catom_lst.append('\[\d+HH\]')
multi_catom_lst.append('\[1H\-\]')
multi_catom_lst.append('\[2H\]')
multi_catom_lst.append('\[2H\-\]')
multi_catom_lst.append('\[3H\]')
multi_catom_lst.append('\[3H\-\]')
multi_catom_lst.append('\[te\]')
multi_catom_lst.append('\*')
# multi_catom_lst.append('\[Cn\]')
# multi_catom_lst.append('\[Np\]')


# print(multi_catom_lst)
# print(single_atom_lst)
# re pattern match
import re

# pattens = 'Cl|c|C'
pattens = '|'.join(multi_catom_lst) + '|' + '|'.join(single_atom_lst) # the front ones have the pirority
# c_smiles = 'C1=C(C(C=C(C1O)Cl)Cl)Cl'

# c_smiles = 'CC1=C(C=CC(=C1)C2=CC(=C(C=C2)N=NC3=C(C4=C(C=C(C=C4C=C3S(=O)(=O)[O-])S(=O)(=O)[O-])N)O)C)N=NC5=C(C6=C(C=C(C=C6C=C5S(=O)(=O)[O-])S(=O)(=O)[O-])N)O.[Na+].[Na+].[Na+].[Na+]'
# atom_lst = re.findall(pattens, c_smiles)

bond_type = ['=', '#', "[\\\\]", "[/]", ':', '[.]', '[$]']


# print(bond_type)
# virtual_str = '=b=$c#:bbb\Aaa/ccc\ccc/..abc'
# print(virtual_str)
special_patterns = '|'.join(bond_type)
# print(re.findall(special_patterns, virtual_str))
# match_idx = [(m.start(0), m.end(0)) for m in re.finditer(special_patterns, virtual_str)]
# print(match_idx)


# assume smiles has been canonicalized
import numpy as np


# labels is generated by the atom split algorithm, length is equal to the atom number 
def parse_smiles_label(smiles, labels):
    s_len = len(smiles)
    smiles_labels = np.array([-1 for _ in range(s_len)])
    atoms_num = len(labels)
    # find atoms:
    match_idx = [(m.start(0), m.end(0)) for m in re.finditer(pattens, smiles)]
    
    
    # give atoms labels
    assert len(match_idx) == atoms_num, "{} is invalid".format(smiles)
    for i in range(atoms_num):
        smiles_labels[match_idx[i][0]: match_idx[i][1]] = labels[i]
    
    
    # iterative the smiles string, give the left special char label
    
    # for ( and [
    for i, ele in enumerate(smiles):
        if ele == '(' or ele == '[':
            j = i
            while smiles_labels[j] == -1:
                j += 1
            smiles_labels[i] = smiles_labels[j]
    
    # for i, ele in enumerate(smiles):
    #     if ele == '(' or ele == '[':
    #         j = i
    #         bracket_label = smiles_labels[i]
    #         if ele == '(':
    #             end_c = ')'
    #         else:
    #             end_c = ']'
    #         # find ')' index
    #         while smiles[j] != end_c:
    #             if smiles_labels[j] != -1:
    #                 bracket_label = smiles_labels[j]
    #             j += 1
    #         for k in range(i, j+1):
    #             smiles_labels[k] = bracket_label
        
    # for the bond symbol, have two label, left atom and right atom label together
    bond_match_idx = [(m.start(0), m.end(0)) for m in re.finditer(special_patterns, smiles)]
    for ele in bond_match_idx:
        # get the bond
        start_idx, end_idx = ele
        assert (end_idx - start_idx) == 1
        
        i = start_idx
        j = start_idx
        while smiles_labels[i] == -1:
            i -= 1
        # while smiles_labels[j] == -1:
        #     j += 1

        # smiles_labels[start_idx] = [smiles_labels[i], smiles_labels[j]]
        smiles_labels[start_idx] = smiles_labels[i]    
    
    
    # for the left uninitialised symbol, give the fonter symbol label
    
    cur_label = -1
    for i, ele in enumerate(smiles_labels):
        
        if smiles_labels[i] != -1:
            cur_label = smiles_labels[i]
        else:
            assert cur_label != -1
            smiles_labels[i] = cur_label
            
    return smiles_labels


def parse_smiles_label_2(smiles, labels):
    s_len = len(smiles)
    smiles_labels = np.array([-1 for _ in range(s_len)])
    atoms_num = len(labels)
    # find atoms:
    match_idx = [(m.start(0), m.end(0)) for m in re.finditer(pattens, smiles)]
    
    
    # give atoms labels
    assert len(match_idx) == atoms_num, "{} is invalid".format(smiles)
    for i in range(atoms_num):
        smiles_labels[match_idx[i][0]: match_idx[i][1]] = labels[i]
    
    
    # iterative the smiles string, give the left special char label
    
    # for ( and [
    for i, ele in enumerate(smiles):
        if ele == '(' or ele == '[':
            j = i
            bracket_label = smiles_labels[i]
            if ele == '(':
                end_c = ')'
            else:
                end_c = ']'
            # find ')' index
            while smiles[j] != end_c:
                if smiles_labels[j] != -1:
                    bracket_label = smiles_labels[j]
                j += 1
            for k in range(i, j+1):
                smiles_labels[k] = bracket_label
        
    # for the bond symbol, have two label, left atom and right atom label together
    bond_match_idx = [(m.start(0), m.end(0)) for m in re.finditer(special_patterns, smiles)]
    for ele in bond_match_idx:
        # get the bond
        start_idx, end_idx = ele
        assert (end_idx - start_idx) == 1
        
        i = start_idx
        j = start_idx
        while smiles_labels[i] == -1:
            i -= 1
        while smiles_labels[j] == -1:
            j += 1

        # smiles_labels[start_idx] = [smiles_labels[i], smiles_labels[j]]
        smiles_labels[start_idx] = smiles_labels[i]
    
    
    # for the left uninitialised symbol, give the fonter symbol label
    
    cur_label = -1
    for i, ele in enumerate(smiles_labels):
        
        if smiles_labels[i] != -1:
            cur_label = smiles_labels[i]
        else:
            assert cur_label != -1
            smiles_labels[i] = cur_label
            
    return smiles_labels




from statistics import mode


def most_common(List):
    return(mode(List))

def get_token_label(smiles_tokenizer, smiles, smiles_labels):
    token_split_res = smiles_tokenizer.tokenize(smiles)

    token_len = len(token_split_res)

    token_labels = np.zeros(token_len)
    start_idx = 0

    for i, t_ele in enumerate(token_split_res):
        token_len = len(t_ele)
        frag_label_array = smiles_labels[start_idx: start_idx+token_len]
        
        # find the most frequency element in frag_label_array
        most_frag_label = most_common(frag_label_array)
        token_labels[i] = most_frag_label
        start_idx += token_len   

    # token labels
    return token_labels
    # print(token_labels)


def ifg_detect_fg_label(smiles, smiles_tokenizer):
    """
    return token labels and graph labels
    """
    mol = Chem.MolFromSmiles(smiles)
    mol_num = len(list(mol.GetAtoms()))
    fgs = identify_functional_groups(mol)
    # parse smiles 
    token_split_res = smiles_tokenizer.tokenize(smiles)
    token_len = len(token_split_res)

    if not len(fgs):
        # no fg detected, return all 0
        return [0 for _ in range(token_len)], [0 for _ in range(mol_num)]
    
    # if have fg, set the non-fg part to -1
    token_labels = [-1 for _ in range(token_len)]
    mol_labels = [-1 for _ in range(mol_num)]

    # set mol labels
    fg_id = 0
    for ele in fgs:
        for atom_id in ele.atomIds:
            mol_labels[atom_id] = fg_id
        fg_id += 1
    
    # parse atoms from tokens
    match_idx = [(m.start(0), m.end(0)) for m in re.finditer(pattens, smiles)]
    
    smiles_labels = np.array([-1 for _ in range(len(smiles))])
    # give atoms labels
    assert len(match_idx) == mol_num, "{} is invalid".format(smiles)
    for i in range(mol_num):
        smiles_labels[match_idx[i][0]: match_idx[i][1]] = mol_labels[i]

    start_idx = 0
    for i, t_ele in enumerate(token_split_res):
        token_len = len(t_ele)
        frag_label_array = smiles_labels[start_idx: start_idx+token_len]
        
        # find the most frequency element in frag_label_array
        token_l = -1
        for f_label in frag_label_array:
            if f_label > -1:
                token_l = f_label # if 2 atoms in the same token, may have discontinuous

        token_labels[i] = token_l
        start_idx += token_len 

    return token_labels, mol_labels



def brics_decomp(mol, addition_rule=False, return_all_bonds=True):
    """
    return break bonds, use additional rule or not
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [], []

    cliques = []
    breaks = []
    all_bonds = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])
        all_bonds.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))

    cut_bonds_set = [list(ele[0]) for ele in res]
    
    
    if not addition_rule:
        if return_all_bonds:
            return cut_bonds_set, all_bonds
        else:
            return cut_bonds_set

    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge breaks
    cut_bonds_set.extend(breaks)
    if return_all_bonds:
        return cut_bonds_set, all_bonds
    else:
        return cut_bonds_set


def brics_decompose_mol(mol, addition_rule=False):
    cut_bonds_set, all_bonds = brics_decomp(mol, addition_rule=addition_rule, return_all_bonds=True)
    g_labels_brics = fragment_graph_cutbonds(mol, cut_bonds_set)
    return g_labels_brics


# frag
def ifg_detect_mol_brics(mol, addition_rule=False, smiles=""):
    fgs = identify_functional_groups(mol)
    fg_mol_lst = []
    # frag_smiles_lst = []
    for fg in fgs:
        atomdids = list(fg.atomIds)
        fg_mol_lst.append(atomdids)
        # frag_smiles = Chem.MolFragmentToSmiles(mol, atomdids, kekuleSmiles=True)
        # frag_smiles_lst.append(frag_smiles)
    
    # print(f'fg list is {fg_mol_lst}\n')

    cut_bonds_set, all_bonds = brics_decomp(mol, addition_rule=addition_rule, return_all_bonds=True)


    g_labels_brics = fragment_graph_cutbonds(mol, cut_bonds_set)
    
    gl_dict = defaultdict(list)
    brics_group_num = 0
    for i, gl in enumerate(g_labels_brics):
        gl_dict[gl].append(i)
    
    brics_group_num = len(gl_dict)
    # print(f'orgin brics dict is {gl_dict}\n')
    # function group number



    # erase cut_bonds_set, if two node of an edge both occur in same function group
    remove_lst = []
    for cut_bond in cut_bonds_set:
        for fg in fg_mol_lst:
            try:
                if cut_bond[0] in fg and cut_bond[1] in fg:
                    remove_lst.append(cut_bond)
            except:
                print(f'{smiles} error')
                import pdb; pdb.set_trace()
                
    for cut_bond in remove_lst:
        cut_bonds_set.remove(cut_bond)
    # print(f'cut_bonds_set is {cut_bonds_set}\n')
    # add to cut_bonds_set new cut_bond
    for bond in all_bonds:
        for fg in fg_mol_lst:
            if (bond[0] in fg and bond[1] not in fg) or (bond[1] in fg and bond[0] not in fg): # keep such bonds
                cut_bonds_set.append(bond)
    # print(f'cut_bonds_set is {cut_bonds_set}\n')

    g_labels = fragment_graph_cutbonds(mol, cut_bonds_set)
    
    return g_labels




if __name__ == "__main__":
    # smiles token
    from iupac_token import SmilesTokenizer

    smiles_tokenizer_path = '/sharefs/sharefs-test_data/iupac-pretrain/smiles_tokenizer'
    smiles_tokenizer = SmilesTokenizer.from_pretrained(smiles_tokenizer_path, max_len=128)
    smiles = 'CCC[C@@H](O)CC\C=C\C=C\C#CC#C\C=C\COCCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO'
    ifg_detect_fg_label(smiles, smiles_tokenizer)
    smiles = '[55Fe+2]'
    mol = Chem.MolFromSmiles(smiles)
    graph_labels = ifg_detect_mol_brics(mol, addition_rule=False, smiles=smiles)

# smiles = 'O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C'
# m = Chem.MolFromSmiles(smiles)
# res = list(Chem.BRICS.BRICSDecompose(m))
# print(sorted(res))


# labels = fragmeng_graph(smiles)
# print(labels)



# # smiles_labels = parse_smiles_label(smiles, labels)
# smiles_labels = parse_smiles_label_2(smiles, labels)
# # print(smiles_labels)
# print(' '.join([ele for ele in smiles]))
# print(' '.join([str(ele) for ele in smiles_labels]))
# token_labels = get_token_label(smiles_tokenizer, smiles, smiles_labels)

# split_tokens = smiles_tokenizer.tokenize(smiles)
# print(' '.join(split_tokens))
# print(' '.join([str(int(ele)) for ele in token_labels]))
# print(token_labels)