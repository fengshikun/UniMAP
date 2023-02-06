from dataclasses import replace
from operator import add
import os
import rdkit.Chem as Chem
import numpy as np
import torch
import random
# !pip install datasets
from datasets import load_dataset
import pandas as pd
# from rdkit import Chem
# from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from torch.utils.data import Dataset

from utils.mol import smiles2graph
from utils.torchvocab import MolVocab

# import pyximport
# pyximport.install(setup_args={"include_dirs": np.get_include()})
# from . import algos

from frag_graph_smiles import get_token_label, parse_smiles_label, fragmeng_graph, ifg_detect_fg_label, brics_decompose_mol, ifg_detect_mol_brics

class RawTextDataset(Dataset):
    """
    Custom Torch Dataset for tokenizing large (up to 100,000,000+ sequences) text corpuses,
    by not loading the entire dataset into cache and using lazy loading from disk (using huggingface's
    'NLP' library. See 'https://github.com/huggingface/nlp' for more details on the NLP package.
    Examples
    --------
    >>> from raw_text_dataset import RawTextDataset
    >>> dataset = RawTextDataset(tokenizer=tokenizer, file_path="shard_00_selfies.txt", block_size=512)
    Downloading: 100%
    1.52k/1.52k [00:03<00:00, 447B/s]
    Using custom data configuration default
    Downloading and preparing dataset text/default-f719ef2eb3ab586b (download: Unknown size, generated: Unknown size, post-processed: Unknown sizetotal: Unknown size) to /root/.cache/huggingface/datasets/text/default-f719ef2eb3ab586b/0.0.0/3a79870d85f1982d6a2af884fde86a71c771747b4b161fd302d28ad22adf985b...
    Dataset text downloaded and prepared to /root/.cache/huggingface/datasets/text/default-f719ef2eb3ab586b/0.0.0/3a79870d85f1982d6a2af884fde86a71c771747b4b161fd302d28ad22adf985b. Subsequent calls will reuse this data.
    Loaded Dataset
    Number of lines: 999988
    Block size: 512
    """

    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("text", data_files=data_files)["train"]
        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __len__(self):
        return self.len

    def preprocess(self, feature_dict):
        # print(feature_dict)
        batch_encoding = self.tokenizer(
            feature_dict['text'],#["Preferred"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.block_size,
        )
        return torch.tensor(batch_encoding["input_ids"])

    def __getitem__(self, i):
        line = self.dataset[i]
        example = self.preprocess(line)
        

        # print(line)
        # print(example)
        return example


class RawTextDatasetDual(Dataset):
    """
    Custom Torch Dataset for tokenizing large (up to 100,000,000+ sequences) text corpuses,
    by not loading the entire dataset into cache and using lazy loading from disk (using huggingface's
    'NLP' library. See 'https://github.com/huggingface/nlp' for more details on the NLP package.
    Examples
    --------
    >>> from raw_text_dataset import RawTextDataset
    >>> dataset = RawTextDataset(tokenizer=tokenizer, file_path="shard_00_selfies.txt", block_size=512)
    Downloading: 100%
    1.52k/1.52k [00:03<00:00, 447B/s]
    Using custom data configuration default
    Downloading and preparing dataset text/default-f719ef2eb3ab586b (download: Unknown size, generated: Unknown size, post-processed: Unknown sizetotal: Unknown size) to /root/.cache/huggingface/datasets/text/default-f719ef2eb3ab586b/0.0.0/3a79870d85f1982d6a2af884fde86a71c771747b4b161fd302d28ad22adf985b...
    Dataset text downloaded and prepared to /root/.cache/huggingface/datasets/text/default-f719ef2eb3ab586b/0.0.0/3a79870d85f1982d6a2af884fde86a71c771747b4b161fd302d28ad22adf985b. Subsequent calls will reuse this data.
    Loaded Dataset
    Number of lines: 999988
    Block size: 512
    """

    def __init__(self, tokenizer, tokenizer_smiles, file_path: str, block_size: int, file_path_smiles: str, block_size_smiles: int, hard_replace=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        
        self.tokenizer_smiles = tokenizer_smiles
        self.file_path_smiles = file_path_smiles
        self.block_size_smiles = block_size_smiles

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("text", data_files=data_files)["train"]
        print("Loaded Dataset")
        self.length = len(self.dataset)
        print("Number of lines: " + str(self.length))
        print("Block size: " + str(self.block_size))
        
        data_files_smiles = get_data_files(file_path_smiles)
        self.dataset_smiles = load_dataset("text", data_files=data_files_smiles)["train"]
        print("Loaded Dataset")
        self.len_smiles = len(self.dataset_smiles)
        print("Number of lines: " + str(self.len_smiles))
        print("Block size: " + str(self.block_size_smiles))
        self.hard_replace = hard_replace
        
    def __len__(self):
        return self.length

    def preprocess(self, feature_dict, tokenizer):
        # print(feature_dict)
        batch_encoding = tokenizer(
            feature_dict['text'],#["Preferred"],
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
        )
        return batch_encoding

    def __getitem__(self, i):
        line = self.dataset[i]
        example = self.preprocess(line, self.tokenizer)
        smiles_line = self.dataset_smiles[i]
        example_smiles = self.preprocess(smiles_line, self.tokenizer_smiles)
        
        if self.hard_replace:
            replace_hard = []
            for i in range(self.hard_replace):
                replace_hard.append(self.tokenizer.tokenize_replace(line['text'], self.block_size))
            
            return example, example_smiles, replace_hard

        # print(line)
        # print(example)
        return example, example_smiles

# raw text iupac dataset, return iupac, positive pair(random pick from other columns), hard negative(generate)
class RawTextDatasetIupac(Dataset):
    """
    RawTextDatasetIupac
    """

    def __init__(self, tokenizer, file_path: str, block_size: int, hard_replace=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        
        

        # data_files = get_data_files(file_path)
        
        df = pd.read_csv(file_path)
        self.dataset = df
        # self.dataset = Dataset.from_pandas(df)
        # self.dataset = load_dataset("text", data_files=data_files)["train"]
        print("Loaded Dataset")
        self.length = len(self.dataset)
        print("Number of lines: " + str(self.length))
        print("Block size: " + str(self.block_size))

        self.hard_replace = hard_replace
        
    def __len__(self):
        return self.length

    def preprocess(self, iupac_str, tokenizer):
        # print(feature_dict)
        batch_encoding = tokenizer(
            iupac_str,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
        )
        return batch_encoding

    def __getitem__(self, i):
        line = self.dataset.iloc[i]

        line_array = line.values
        sample = line_array[0] # preferred
        random_pos = random.choice(line_array[1:])

        example = self.preprocess(sample, self.tokenizer)
        example_pos = self.preprocess(random_pos, self.tokenizer)
        
        if self.hard_replace:
            replace_hard = []
            for i in range(self.hard_replace):
                replace_hard.append(self.tokenizer.tokenize_replace(sample, self.block_size))
            
            return example, example_pos, replace_hard

        # print(line)
        # print(example)
        return example, example_pos



class SmilesIUPACDatasetDual(Dataset):
    """
    Dataset to load smiles, iupac and graph, pass the tokenizer that can both tokenize the smiles and iupac
    """

    def __init__(self, tokenizer, file_path: str, block_size: int, atom_vocab_file: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("text", data_files=data_files)["train"]
        print("Loaded Dataset")
        self.length = len(self.dataset)
        print("Number of lines: " + str(self.length))
        print("Block size: " + str(self.block_size))
        
        self.atom_vocab = MolVocab.load_vocab(atom_vocab_file)
        self.atom_vocab_size = len(self.atom_vocab)
        
        
    def __len__(self):
        return self.length

    def preprocess(self, feature_dict, tokenizer):
        # print(feature_dict)
        
        smiles_iupac = feature_dict['text']
        smiles, iupac = smiles_iupac.split('\t')
        
        # generate graph
        graph = smiles2graph(smiles, self.atom_vocab)
        
        tokenizer.set_inpuac_input(False)
        batch_encoding_smiles = tokenizer(
            smiles,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
        )
        tokenizer.set_inpuac_input(True)
        batch_encoding_iupac = tokenizer(
            iupac,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
        )
        return batch_encoding_smiles, batch_encoding_iupac, graph

    def __getitem__(self, i):
        smiles_iupac_line = self.dataset[i]
        batch_encoding_smiles, batch_encoding_iupac, graph = self.preprocess(smiles_iupac_line, self.tokenizer)
        
        
        return batch_encoding_smiles, batch_encoding_iupac, graph


class SmilesIUPACDatasetDualSplit(Dataset):
    """
    Dataset to load smiles, iupac and graph, pass the tokenizer that can both tokenize the smiles and iupac
    """

    def __init__(self, tokenizer_iupac, tokenizer_smiles, file_path: str, block_size: int, atom_vocab_file: str):
        super().__init__()
        self.tokenizer_iupac = tokenizer_iupac
        self.tokenizer_smiles = tokenizer_smiles
        self.file_path = file_path
        self.block_size = block_size
        

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("text", data_files=data_files)["train"]
        print("Loaded Dataset")
        self.length = len(self.dataset)
        print("Number of lines: " + str(self.length))
        print("Block size: " + str(self.block_size))
        
        self.atom_vocab = MolVocab.load_vocab(atom_vocab_file)
        self.atom_vocab_size = len(self.atom_vocab)
        
        
    def __len__(self):
        return self.length

    def preprocess(self, feature_dict):
        # print(feature_dict)
        
        smiles_iupac = feature_dict['text']
        smiles, iupac = smiles_iupac.split('\t')
        
        # generate graph
        graph = smiles2graph(smiles, self.atom_vocab)
        
        batch_encoding_smiles = self.tokenizer_smiles(
            smiles,
            add_special_tokens=True,
            truncation=True,
            # padding="max_length",
            max_length=self.block_size,
        )
        batch_encoding_iupac = self.tokenizer_iupac(
            iupac,
            add_special_tokens=True,
            truncation=True,
            # padding="max_length",
            max_length=self.block_size,
        )
        return batch_encoding_smiles, batch_encoding_iupac, graph

    def __getitem__(self, i):
        smiles_iupac_line = self.dataset[i]
        batch_encoding_smiles, batch_encoding_iupac, graph = self.preprocess(smiles_iupac_line)
        
        
        return batch_encoding_smiles, batch_encoding_iupac, graph



class RegressionTextIterable(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        print("Initializing dataset...")
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        print("Inferring CSV structure from first line...")
        self.dataset = load_dataset("text", data_files=get_data_files(file_path))[
            "train"
        ]
        self.num_labels = len(self.dataset[0]["text"].split(",")) - 1

        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __iter__(self):
        for example in self.dataset:
            yield preprocess(example["text"], self.tokenizer, self.block_size)


class RegressionDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        print("init dataset")
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("csv", data_files=data_files)["train"]
        dataset_columns = list(self.dataset.features.keys())
        self.smiles_column = dataset_columns[0]
        self.label_columns = dataset_columns[1:]
        self.num_labels = len(self.label_columns)

        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return preprocess(self.dataset[i]["text"], self.tokenizer, self.block_size)


class RegressionTextDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        print("Initializing dataset...")
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        print("Inferring CSV structure from first line...")
        self.dataset = load_dataset("text", data_files=get_data_files(file_path))[
            "train"
        ]
        self.num_labels = len(self.dataset[0]["text"].split(",")) - 1
        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return preprocess(self.dataset[i]["text"], self.tokenizer, self.block_size)


def preprocess(line, tokenizer, block_size):
    def _clean_property(x):
        if x == "" or "inf" in x:
            return 0.0
        return float(x)

    line = line.split(",")
    smiles = line[0]
    labels = line[1:]

    batch_encoding = tokenizer(
        smiles,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=block_size,
    )
    batch_encoding["label"] = [_clean_property(x) for x in labels]
    batch_encoding = {k: torch.tensor(v) for k, v in batch_encoding.items()}

    return batch_encoding

'''
class LazyRegressionDataset(Dataset):
    """Computes RDKit properties on-the-fly."""

    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        print("init dataset")
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        self.descriptors = [name for name, _ in Chem.Descriptors.descList]
        self.descriptors.remove("Ipc")
        self.calculator = MolecularDescriptorCalculator(self.descriptors)
        self.num_labels = len(self.descriptors)

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("text", data_files=data_files)["train"]

        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __len__(self):
        return self.len

    def _compute_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol_descriptors = np.full(shape=(self.num_labels), fill_value=0.0)
        else:
            mol_descriptors = np.array(list(self.calculator.CalcDescriptors(mol)))
            mol_descriptors = np.nan_to_num(
                mol_descriptors, nan=0.0, posinf=0.0, neginf=0.0
            )
        assert mol_descriptors.size == self.num_labels

        return mol_descriptors

    def preprocess(self, feature_dict):
        smiles = feature_dict["text"]
        batch_encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
        )
        batch_encoding = {k: torch.tensor(v) for k, v in batch_encoding.items()}

        mol_descriptors = self._compute_descriptors(smiles)
        batch_encoding["label"] = torch.tensor(mol_descriptors, dtype=torch.float32)

        return batch_encoding

    def __getitem__(self, i):
        feature_dict = self.dataset[i]
        example = self.preprocess(feature_dict)
        return example
'''

def get_data_files(train_path):
    if os.path.isdir(train_path):
        return [
            os.path.join(train_path, file_name) for file_name in os.listdir(train_path)
        ]
    elif os.path.isfile(train_path):
        return train_path

    raise ValueError("Please pass in a proper train path")



class SmilesIUPACDatasetDualSplitUNI(Dataset):
    """
    Dataset to load smiles, iupac and graph, pass the tokenizer that can both tokenize the smiles and iupac
    """

    def __init__(self, tokenizer_iupac, tokenizer_smiles, file_path: str, block_size: int, atom_vocab_file: str, \
        function_group_file: str, finger_print_file: str, replace_prob=0.5, replace_graph_prob=0.5, replace_iupac_prob=0, smiles_only=False, get_frag=False, detect_fg=False, mix_brics=False, brics_plus=False, max_atom_num=156):
        super().__init__()
        self.tokenizer_iupac = tokenizer_iupac
        self.tokenizer_smiles = tokenizer_smiles
        self.file_path = file_path
        self.block_size = block_size
        

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("text", data_files=data_files)["train"]
        print("Loaded Dataset")
        self.length = len(self.dataset)
        print("Number of lines: " + str(self.length))
        print("Block size: " + str(self.block_size))
        
        self.atom_vocab = MolVocab.load_vocab(atom_vocab_file)
        self.atom_vocab_size = len(self.atom_vocab)

        # load function group and finger-print labels
        self.function_group_labels = np.load(function_group_file, mmap_mode="r")
        self.finger_print_labels = np.load(finger_print_file, mmap_mode="r")
        
        self.replace_prob = replace_prob
        self.replace_graph_prob = replace_graph_prob
        self.replace_iupac_prob = replace_iupac_prob
        assert replace_graph_prob + replace_iupac_prob == replace_prob
        
        
        
        self.get_frag = get_frag
        
        self.smiles_only = smiles_only
        if self.smiles_only: # no iupac
            assert replace_iupac_prob == 0
        
        self.detect_fg = detect_fg
        self.mix_brics = mix_brics
        self.brics_plus = brics_plus
        
        self.max_atom_num = max_atom_num
        # for debug
        # self.cnt = 1
        
    def __len__(self):
        return self.length

    def preprocess(self, feature_dict, get_frag=False):
        # print(feature_dict)
        
        smiles_iupac = feature_dict['text']
        if self.smiles_only:
            smiles = smiles_iupac
        else:
            smiles, iupac = smiles_iupac.split('\t')
        
        
        
        # generate graph
        graph = smiles2graph(smiles, self.atom_vocab)
        
        if get_frag:
            mol = Chem.MolFromSmiles(smiles)
            if self.detect_fg:
                token_labels, graph_labels = ifg_detect_fg_label(smiles, self.tokenizer_smiles)
            else:
                if self.mix_brics:
                    if self.brics_plus:
                        graph_labels = ifg_detect_mol_brics(mol, addition_rule=True)
                    else:
                        graph_labels = ifg_detect_mol_brics(mol, addition_rule=False, smiles=smiles)
                else:
                    if self.brics_plus:
                        graph_labels = brics_decompose_mol(mol, addition_rule=True)
                    else:
                        graph_labels = fragmeng_graph(smiles)
                

                # debug
                # frag_smiles_lst = []
                # group_num = max(graph_labels) + 1
                # for g_id in range(group_num):
                #     g_mol_lst = []
                #     for i, l in enumerate(graph_labels):
                #         if l == g_id:
                #             g_mol_lst.append(i)

                #     frag_smiles = Chem.MolFragmentToSmiles(mol, g_mol_lst, kekuleSmiles=True)
                #     frag_smiles_lst.append(frag_smiles)
                # print(f'{frag_smiles_lst}')

                # # debug
                # exit(0)

                smiles_labels = parse_smiles_label(smiles, graph_labels)
                token_labels = get_token_label(self.tokenizer_smiles, smiles, smiles_labels)
            
            # for debug
            if sum(token_labels) == 0:
                pass
                # print('xxxx')
            max_number = int(max(token_labels) + 1)
            for i in range(max_number):
                if i not in token_labels:
                    pass
                    # print('xxxxx')
        
        batch_encoding_smiles = self.tokenizer_smiles(
            smiles,
            add_special_tokens=True,
            truncation=True,
            # padding="max_length",
            max_length=self.block_size,
        )
        
        if self.smiles_only:
            batch_encoding_iupac = None
        else:
            batch_encoding_iupac = self.tokenizer_iupac(
                iupac,
                add_special_tokens=True,
                truncation=True,
                # padding="max_length",
                max_length=self.block_size,
            )
        
        if get_frag:
            return batch_encoding_smiles, batch_encoding_iupac, graph, graph_labels, token_labels
        
        return batch_encoding_smiles, batch_encoding_iupac, graph
    
    def _get_random_idx(self, exclude_idx):
        for _ in range(100):
            rand_doc_idx = random.randrange(0, self.length)
            if rand_doc_idx != exclude_idx:
                break
        return rand_doc_idx

    def __getitem__(self, i):
        smiles_iupac_line = self.dataset[i]

        # for debug:
        # smiles_iupac_line = {'text': 'C1=c2cc[nH]c2=C(c2ccc(-c3ccccc3)cc2)CC1c1ccc(-c2ccccc2)cc1'}

        # check atom number
        smiles_string = smiles_iupac_line['text']
        mol = Chem.MolFromSmiles(smiles_string)
        atom_num = mol.GetNumAtoms()
        while atom_num > self.max_atom_num:
            new_idx = self._get_random_idx(i)
            smiles_iupac_line = self.dataset[new_idx]
            # check atom number
            smiles_string = smiles_iupac_line['text']
            mol = Chem.MolFromSmiles(smiles_string)
            atom_num = mol.GetNumAtoms()
            i = new_idx



        # print("cnt: {}, smiles is {}".format(self.cnt, smiles_iupac_line))
        # self.cnt += 1
        
        replace_graph = False
        replace_iupac = False
        
        if self.get_frag:
            batch_encoding_smiles, batch_encoding_iupac, graph, graph_labels, token_labels = self.preprocess(smiles_iupac_line, True)
        else:
            batch_encoding_smiles, batch_encoding_iupac, graph = self.preprocess(smiles_iupac_line)
        rand_dice = random.random()
        ctr_label = 0
        if rand_dice < self.replace_prob:
            idx_j = self._get_random_idx(i)
            smiles_iupac_line_j = self.dataset[idx_j]
            smiles_string = smiles_iupac_line_j['text']
            mol = Chem.MolFromSmiles(smiles_string)
            atom_num = mol.GetNumAtoms()
            while atom_num > self.max_atom_num:
                new_idx = self._get_random_idx(idx_j)
                smiles_iupac_line_j = self.dataset[new_idx]
                # check atom number
                smiles_string = smiles_iupac_line_j['text']
                mol = Chem.MolFromSmiles(smiles_string)
                atom_num = mol.GetNumAtoms()
                idx_j = new_idx


            
            if rand_dice > self.replace_iupac_prob: # replace graph
                replace_graph = True
                _, _, replace_graph = self.preprocess(smiles_iupac_line_j)
                ctr_label = 1
            else: # replace iupac
                replace_iupac = True
                _, replace_batch_encoding_iupac, _ = self.preprocess(smiles_iupac_line_j)
                ctr_label = 2
        
        # finger print maybe nan
        replace_token = 0
        finger_print_label = np.where(np.isnan(self.finger_print_labels[i]), replace_token, self.finger_print_labels[i])


        if self.get_frag:
            if replace_graph:
                graph = [graph, replace_graph]
            if replace_iupac:
                batch_encoding_iupac = [batch_encoding_iupac, replace_batch_encoding_iupac]

            if self.get_frag:
                return batch_encoding_smiles, batch_encoding_iupac, graph, ctr_label,\
                np.float32(self.function_group_labels[i]), np.float32(finger_print_label), graph_labels, token_labels, smiles_iupac_line['text']
        else:
            graph = replace_graph
            batch_encoding_iupac = replace_batch_encoding_iupac
        
            return batch_encoding_smiles, batch_encoding_iupac, graph, ctr_label,\
                np.float32(self.function_group_labels[i]), np.float32(finger_print_label)


class SmilesIUPACDatasetDualSplitUNIG(Dataset):
    """
    Dataset to load smiles, iupac and graph, pass the tokenizer that can both tokenize the smiles and iupac
    """

    def __init__(self, tokenizer_iupac, tokenizer_smiles, file_path: str, block_size: int, atom_vocab_file: str, \
        function_group_file: str, finger_print_file: str, replace_prob=0.5, replace_graph_prob=0.5, replace_iupac_prob=0, smiles_only=False, get_frag=False):
        super().__init__()
        self.tokenizer_iupac = tokenizer_iupac
        self.tokenizer_smiles = tokenizer_smiles
        self.file_path = file_path
        self.block_size = block_size
        

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("text", data_files=data_files)["train"]
        print("Loaded Dataset")
        self.length = len(self.dataset)
        print("Number of lines: " + str(self.length))
        print("Block size: " + str(self.block_size))
        
        self.atom_vocab = MolVocab.load_vocab(atom_vocab_file)
        self.atom_vocab_size = len(self.atom_vocab)

        # load function group and finger-print labels
        self.function_group_labels = np.load(function_group_file, mmap_mode="r")
        self.finger_print_labels = np.load(finger_print_file, mmap_mode="r")
        
        self.replace_prob = replace_prob
        self.replace_graph_prob = replace_graph_prob
        self.replace_iupac_prob = replace_iupac_prob
        assert replace_graph_prob + replace_iupac_prob == replace_prob
        
        
        
        self.get_frag = get_frag
        
        self.smiles_only = smiles_only
        if self.smiles_only: # no iupac
            assert replace_iupac_prob == 0
            
        # for debug
        # self.cnt = 1
        
    def __len__(self):
        return self.length

    def preprocess_item(self, item):
        edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
        N = x.size(0)
        # x = self.convert_to_single_emb(x)

        # node adj matrix [N, N] bool
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        # edge feature here
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        # attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        #     self.convert_to_single_emb(edge_attr) + 1
        # )

        
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

        # combine
        # item.x = x
        info = {}
        info['attn_bias'] = attn_bias
        # item.attn_edge_type = attn_edge_type
        info['spatial_pos'] = spatial_pos
        info['in_degree'] = adj.long().sum(dim=1).view(-1)
        # info['out_degree'] = item.in_degree  # for undirected graph
        info['edge_input'] = torch.from_numpy(edge_input).long()

        return info

    def preprocess(self, feature_dict, get_frag=False):
        # print(feature_dict)
        
        smiles_iupac = feature_dict['text']
        if self.smiles_only:
            smiles = smiles_iupac
        else:
            smiles, iupac = smiles_iupac.split('\t')
        
        
        
        # generate graph
        graph = smiles2graph(smiles, self.atom_vocab)
        info = self.preprocess_item(graph[0])
        graph = graph + (info,)
        
        if get_frag:
            graph_labels = fragmeng_graph(smiles)
            smiles_labels = parse_smiles_label(smiles, graph_labels)
            token_labels = get_token_label(self.tokenizer_smiles, smiles, smiles_labels)
            
            # for debug
            if sum(token_labels) == 0:
                pass
                # print('xxxx')
            max_number = int(max(token_labels) + 1)
            for i in range(max_number):
                if i not in token_labels:
                    pass
                    # print('xxxxx')
        
        batch_encoding_smiles = self.tokenizer_smiles(
            smiles,
            add_special_tokens=True,
            truncation=True,
            # padding="max_length",
            max_length=self.block_size,
        )
        
        if self.smiles_only:
            batch_encoding_iupac = None
        else:
            batch_encoding_iupac = self.tokenizer_iupac(
                iupac,
                add_special_tokens=True,
                truncation=True,
                # padding="max_length",
                max_length=self.block_size,
            )
        
        if get_frag:
            return batch_encoding_smiles, batch_encoding_iupac, graph, graph_labels, token_labels
        
        return batch_encoding_smiles, batch_encoding_iupac, graph
    
    def _get_random_idx(self, exclude_idx):
        for _ in range(100):
            rand_doc_idx = random.randrange(0, self.length)
            if rand_doc_idx != exclude_idx:
                break
        return rand_doc_idx

    def __getitem__(self, i):
        smiles_iupac_line = self.dataset[i]
        
        # print("cnt: {}, smiles is {}".format(self.cnt, smiles_iupac_line))
        # self.cnt += 1
        
        replace_graph = False
        replace_iupac = False
        
        if self.get_frag:
            batch_encoding_smiles, batch_encoding_iupac, graph, graph_labels, token_labels = self.preprocess(smiles_iupac_line, True)
        else:
            batch_encoding_smiles, batch_encoding_iupac, graph = self.preprocess(smiles_iupac_line)
        rand_dice = random.random()
        ctr_label = 0
        if rand_dice < self.replace_prob:
            idx_j = self._get_random_idx(i)
            smiles_iupac_line_j = self.dataset[idx_j]
            if rand_dice > self.replace_iupac_prob: # replace graph
                replace_graph = True
                _, _, replace_graph = self.preprocess(smiles_iupac_line_j)
                ctr_label = 1
            else: # replace iupac
                replace_iupac = True
                _, replace_batch_encoding_iupac, _ = self.preprocess(smiles_iupac_line_j)
                ctr_label = 2
        
        # finger print maybe nan
        replace_token = 0
        finger_print_label = np.where(np.isnan(self.finger_print_labels[i]), replace_token, self.finger_print_labels[i])


        if self.get_frag:
            if replace_graph:
                graph = [graph, replace_graph]
            if replace_iupac:
                batch_encoding_iupac = [batch_encoding_iupac, replace_batch_encoding_iupac]

            if self.get_frag:
                return batch_encoding_smiles, batch_encoding_iupac, graph, ctr_label,\
                np.float32(self.function_group_labels[i]), np.float32(finger_print_label), graph_labels, token_labels
        else:
            graph = replace_graph
            batch_encoding_iupac = replace_batch_encoding_iupac
        
            return batch_encoding_smiles, batch_encoding_iupac, graph, ctr_label,\
                np.float32(self.function_group_labels[i]), np.float32(finger_print_label)




class SmilesIUPACDatasetDualSplitUNIEval(Dataset):
    """
    Dataset to load smiles, iupac and graph, pass the tokenizer that can both tokenize the smiles and iupac
    """

    def __init__(self, tokenizer_iupac, tokenizer_smiles, file_path: str, block_size: int, atom_vocab_file: str, \
        function_group_file: str, finger_print_file: str, replace_prob=0.5, replace_graph_prob=0.5, replace_iupac_prob=0, smiles_only=False, get_frag=False):
        super().__init__()
        self.tokenizer_iupac = tokenizer_iupac
        self.tokenizer_smiles = tokenizer_smiles
        self.file_path = file_path
        self.block_size = block_size
        

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("text", data_files=data_files)["train"]
        print("Loaded Dataset")
        self.length = len(self.dataset)
        print("Number of lines: " + str(self.length))
        print("Block size: " + str(self.block_size))
        
        self.atom_vocab = MolVocab.load_vocab(atom_vocab_file)
        self.atom_vocab_size = len(self.atom_vocab)

        # load function group and finger-print labels
        self.function_group_labels = np.load(function_group_file, mmap_mode="r")
        self.finger_print_labels = np.load(finger_print_file, mmap_mode="r")
        
        self.replace_prob = replace_prob
        self.replace_graph_prob = replace_graph_prob
        self.replace_iupac_prob = replace_iupac_prob
        assert replace_graph_prob + replace_iupac_prob == replace_prob
        
        
        
        self.get_frag = get_frag
        
        self.smiles_only = smiles_only
        if self.smiles_only: # no iupac
            assert replace_iupac_prob == 0
            
        # for debug
        self.cnt = 1
        
    def __len__(self):
        return self.length

    def preprocess(self, feature_dict, get_frag=False):
        # print(feature_dict)
        
        smiles_iupac = feature_dict['text']
        if self.smiles_only:
            smiles = smiles_iupac
        else:
            smiles, iupac = smiles_iupac.split('\t')
        
        
        
        # generate graph
        graph = smiles2graph(smiles, self.atom_vocab)
        
        if get_frag:
            graph_labels = fragmeng_graph(smiles)
            smiles_labels = parse_smiles_label(smiles, graph_labels)
            token_labels = get_token_label(self.tokenizer_smiles, smiles, smiles_labels)
            
            # for debug
            if sum(token_labels) == 0:
                pass
                # print('xxxx')
            max_number = int(max(token_labels) + 1)
            for i in range(max_number):
                if i not in token_labels:
                    pass
                    # print('xxxxx')
        
        batch_encoding_smiles = self.tokenizer_smiles(
            smiles,
            add_special_tokens=True,
            truncation=True,
            # padding="max_length",
            max_length=self.block_size,
        )
        
        if self.smiles_only:
            batch_encoding_iupac = None
        else:
            batch_encoding_iupac = self.tokenizer_iupac(
                iupac,
                add_special_tokens=True,
                truncation=True,
                # padding="max_length",
                max_length=self.block_size,
            )
        
        if get_frag:
            return batch_encoding_smiles, batch_encoding_iupac, graph, graph_labels, token_labels
        
        return batch_encoding_smiles, batch_encoding_iupac, graph
    
    def _get_random_idx(self, exclude_idx):
        for _ in range(100):
            rand_doc_idx = random.randrange(0, self.length)
            if rand_doc_idx != exclude_idx:
                break
        return rand_doc_idx

    def __getitem__(self, i):
        smiles_iupac_line = self.dataset[i]
        
        # print("cnt: {}, smiles is {}".format(self.cnt, smiles_iupac_line))
        self.cnt += 1
        
        # replace_graph = False
        # replace_iupac = False
        
        if self.get_frag:
            batch_encoding_smiles, batch_encoding_iupac, graph, graph_labels, token_labels = self.preprocess(smiles_iupac_line, True)
        else:
            batch_encoding_smiles, batch_encoding_iupac, graph = self.preprocess(smiles_iupac_line)
        ctr_label = 0
        # rand_dice = random.random()
        # if rand_dice < self.replace_prob:
        #     idx_j = self._get_random_idx(i)
        #     smiles_iupac_line_j = self.dataset[idx_j]
        #     if rand_dice > self.replace_iupac_prob: # replace graph
        #         replace_graph = True
        #         _, _, replace_graph = self.preprocess(smiles_iupac_line_j)
        #         ctr_label = 1
        #     else: # replace iupac
        #         replace_iupac = True
        #         _, replace_batch_encoding_iupac, _ = self.preprocess(smiles_iupac_line_j)
        #         ctr_label = 2
        
        # finger print maybe nan
        replace_token = 0
        finger_print_label = np.where(np.isnan(self.finger_print_labels[i]), replace_token, self.finger_print_labels[i])

        if self.get_frag:
            return batch_encoding_smiles, batch_encoding_iupac, graph, ctr_label,\
                np.float32(self.function_group_labels[i]), np.float32(finger_print_label), graph_labels, token_labels
        else:
            return batch_encoding_smiles, batch_encoding_iupac, graph, ctr_label,\
                np.float32(self.function_group_labels[i]), np.float32(finger_print_label)

        # if self.get_frag:
        #     if replace_graph:
        #         graph = [graph, replace_graph]
        #     if replace_iupac:
        #         batch_encoding_iupac = [batch_encoding_iupac, replace_batch_encoding_iupac]

        #     if self.get_frag:
        #         return batch_encoding_smiles, batch_encoding_iupac, graph, ctr_label,\
        #         np.float32(self.function_group_labels[i]), np.float32(finger_print_label), graph_labels, token_labels
        # else:
        #     graph = replace_graph
        #     batch_encoding_iupac = replace_batch_encoding_iupac
        
        #     return batch_encoding_smiles, batch_encoding_iupac, graph, ctr_label,\
        #         np.float32(self.function_group_labels[i]), np.float32(finger_print_label)
def collate_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res
