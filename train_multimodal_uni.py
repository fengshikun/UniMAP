""" Script for training a Roberta Masked-Language Model

Usage [SMILES tokenizer]:
    python train_roberta_mlm.py --dataset_path=<DATASET_PATH> --output_dir=<OUTPUT_DIR> --run_name=<RUN_NAME> --tokenizer_type=smiles --tokenizer_path="seyonec/SMILES_tokenized_PubChem_shard00_160k"

Usage [BPE tokenizer]:
    python train_roberta_mlm.py --dataset_path=<DATASET_PATH> --output_dir=<OUTPUT_DIR> --run_name=<RUN_NAME> --tokenizer_type=bpe
"""
import os
from re import A
from absl import app
from absl import flags

# import os
# os.environ["NCCL_DEBUG"] = "INFO"

import transformers
from transformers.trainer_callback import EarlyStoppingCallback

import torch
from torch.utils.data import random_split

import wandb
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM

from utils.raw_text_dataset import SmilesIUPACDatasetDualSplitUNI, collate_tokens

from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from trainner import Trainer


from models import MultilingualModelUNI
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass, field
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from iupac_token import IUPACTokenizer, SmilesIUPACTokenizer, SmilesTokenizer
from transformers import BertTokenizer
import random


from utils.features import get_mask_atom_feature, get_bond_mask_feature
from torch_geometric.data import Batch
import math
import numpy as np

FLAGS = flags.FLAGS

#Choice of experiments
flags.DEFINE_bool(name="do_mlm", default=True, help="whether to pretrain the roberta MLM, if False, you may only train the tokenizer")

# RobertaConfig params
flags.DEFINE_integer(name="vocab_size", default=1000, help="") # smiles and iupac tokenizer token size: 1228
flags.DEFINE_integer(name="max_position_embeddings", default=515, help="") # This needs to be longer than max_tokenizer_len. max_len is currently 514 in seyonec/SMILES_tokenized_PubChem_shard00_160k
# flags.DEFINE_integer(name="num_attention_heads", default=1, help="")
# flags.DEFINE_integer(name="num_hidden_layers", default=1, help="")
flags.DEFINE_integer(name="type_vocab_size", default=1, help="") # for nsp pretraining task, set this to 2
flags.DEFINE_bool(name="fp16", default=True, help="Mixed precision.")

flags.DEFINE_bool(name="resume", default=False, help="Mixed precision.")

# Tokenizer params
flags.DEFINE_enum(name="tokenizer_type", default="BPE", enum_values=["smiles", "bpe", "SMILES", "BPE"], help="")
flags.DEFINE_string(name="tokenizer_path", default="", help="")
flags.DEFINE_integer(name="BPE_min_frequency", default=2, help="")
flags.DEFINE_integer(name="max_tokenizer_len", default=128, help="") # chemBerta: 512
flags.DEFINE_integer(name="tokenizer_block_size", default=128, help="")

flags.DEFINE_integer(name="max_lingua_len", default=256, help="") # chemBerta: 512

# Smiles tokenizer params
flags.DEFINE_string(name="smiles_tokenizer_path", default="", help="")
flags.DEFINE_integer(name="smiles_max_tokenizer_len", default=128, help="") # chemBerta: 512
flags.DEFINE_integer(name="smiles_tokenizer_block_size", default=128, help="")

# Dataset params
flags.DEFINE_string(name="dataset_path", default='/sharefs//chem_data/pubchem/data_10/iupacs.csv', help="")
flags.DEFINE_string(name="output_dir", default="/sharefs//chem_data/results/output_dir/", help="")
flags.DEFINE_string(name="run_name", default="debug_10epochs", help="this argument is needed when mlm training") # iupacs_


# MLM params
flags.DEFINE_float(name="mlm_probability", default=0.20, lower_bound=0.0, upper_bound=1.0, help="")
flags.DEFINE_float(name="mlm_group_probability", default=0.60, lower_bound=0.0, upper_bound=1.0, help="")


# contrastive param
flags.DEFINE_string(name="pooler_type", default="cls", help="")




# Train params
flags.DEFINE_float(name="frac_train", default=0.95, help="")
flags.DEFINE_integer(name="eval_steps", default=1000, help="")
flags.DEFINE_integer(name="logging_steps", default=100, help="")
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_integer(name="num_train_epochs", default=10, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=32, help="")
flags.DEFINE_integer(name="per_device_eval_batch_size", default=32, help="")
flags.DEFINE_integer(name="save_steps", default=10000, help="")
flags.DEFINE_integer(name="save_total_limit", default=10, help="")




flags.mark_flag_as_required("dataset_path")

# dist parameters
flags.DEFINE_integer(name="local_rank", default=-1,
                        help="local_rank for distributed training on gpus")

flags.DEFINE_bool(name="no_cuda", default=False, help="no cuda flags")
flags.DEFINE_bool(name="distributed", default=False, help="")
flags.DEFINE_integer(name="num_gpus", default=0, help="")

# gcn parameters
flags.DEFINE_integer(name="gnn_number_layer", default=3, help="")
flags.DEFINE_float(name="gnn_dropout", default=0.1, help="")
flags.DEFINE_bool(name="conv_encode_edge", default=True, help="")
flags.DEFINE_integer(name="gnn_embed_dim", default=384, help="")
flags.DEFINE_string(name="gnn_aggr", default="maxminmean", help="")
flags.DEFINE_string(name="gnn_norm", default="layer", help="")
flags.DEFINE_string(name="gnn_act", default="gelu", help="")



flags.DEFINE_integer(name="graph_max_seq_size", default=56, help="")


# gcn mask parameter
# vocab file
flags.DEFINE_string(name="atom_vocab_file", default="/data2/test_data/iupac-pretrain/Datas/data_1m/data_1m_atom_vocab.pkl", help="")
flags.DEFINE_integer(name="atom_vocab_size", default=4199, help="")
flags.DEFINE_string(name="function_group_file", default="/data2/test_data/iupac-pretrain/fg_labels.npy", help="")
flags.DEFINE_string(name="finger_print_file", default="/data2/test_data/iupac-pretrain/fingerprint_labels_regression.npy", help="")

flags.DEFINE_float(name="replace_prob", default=0.5, help="")
flags.DEFINE_float(name="replace_graph_prob", default=0.5, help="")
flags.DEFINE_float(name="replace_iupac_prob", default=0, help="")

flags.DEFINE_bool(name="smiles_only", default=False, help="")
flags.DEFINE_bool(name="get_frag", default=False, help="")
flags.DEFINE_bool(name="detect_fg", default=False, help="")
flags.DEFINE_bool(name="mix_brics", default=False, help="")
flags.DEFINE_bool(name="brics_plus", default=False, help="")

flags.DEFINE_bool(name="check_frag", default=False, help="")

# flags.mark_flag_as_required("run_name")
# flags.mark_flag_as_required("tokenizer_type")


# Data collator
@dataclass
class OurDataCollatorWithPadding:
    tokenizer: BertTokenizer
    smiles_tokenizer: BertTokenizer = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    mlm: bool = True
    mlm_probability: float = 0.20
    mlm_group_probability: float = 0.60
    align_batch: bool = False
    mask_batch: bool = True
    switch_num: int = 0


    def _concat_lang(self, lingua_inputs, max_length=156, iupac_start_emb = 587):
        return_dict = {}
        batch_size = lingua_inputs['iupac']['input_ids'].shape[0]
        return_dict['input_ids'] = torch.full((batch_size, max_length), 1) # padding: 1
        return_dict['attention_mask'] = torch.full((batch_size, max_length), 0) # attention mask, default: 0
        
        
        if 'mlm_input_ids' in lingua_inputs['iupac']:
            return_dict['mlm_input_ids'] = torch.full((batch_size, max_length), 1) # padding: 1
            return_dict['mlm_labels'] = torch.full((batch_size, max_length), -100)
        
        for i, smile_len in enumerate(lingua_inputs['smiles_len']):
            iupac_len = lingua_inputs['iupac_len'][i]
            return_dict['input_ids'][i,:smile_len] = lingua_inputs['smiles']['input_ids'][i,:smile_len]
            
            iupac_input_ids = lingua_inputs['iupac']['input_ids'][i,1:iupac_len] # earse the iupac cls token
            iupac_input_ids[:-1] += iupac_start_emb # except the sep token
            return_dict['input_ids'][i,smile_len: smile_len + iupac_len - 1] = iupac_input_ids
            
            return_dict['attention_mask'][i, :smile_len] = lingua_inputs['smiles']['attention_mask'][i,:smile_len]
            return_dict['attention_mask'][i,smile_len: smile_len + iupac_len - 1] = lingua_inputs['iupac']['attention_mask'][i,1:iupac_len] # erase the iupac cls token
        
            if 'mlm_input_ids' in lingua_inputs['iupac']:
                return_dict['mlm_input_ids'][i, :smile_len] = lingua_inputs['smiles']['mlm_input_ids'][i,:smile_len]
                
                iupac_mlm_input_ids = lingua_inputs['iupac']['mlm_input_ids'][i,1:iupac_len]
                iupac_mlm_input_ids[iupac_mlm_input_ids != 4] += iupac_start_emb
                return_dict['mlm_input_ids'][i,smile_len: smile_len + iupac_len - 1] =iupac_mlm_input_ids # earse the iupac cls token

                return_dict['mlm_labels'][i, :smile_len] = lingua_inputs['smiles']['mlm_labels'][i,:smile_len]
                
                iupac_mlm_labels = lingua_inputs['iupac']['mlm_labels'][i,1:iupac_len]
                iupac_mlm_labels[iupac_mlm_labels != -100] += iupac_start_emb
            
                return_dict['mlm_labels'][i,smile_len: smile_len + iupac_len - 1] = iupac_mlm_labels # earse the iupac cls token
                
        
        return return_dict


    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        # split features to iupac input_ids, smiles input_ids, hard iupac
        # if self.smiles_tokenizer is not None:
        #     lingua_inputs = {'iupac': [], 'smiles': [], 'iupac_len':[], 'smiles_len': []}
        # else:
        #     lingua_inputs = []
        lingua_inputs = {'iupac': [], 'smiles': [], 'iupac_len':[], 'smiles_len': []}
        graph_inputs = []
        gm_labels = []
        ctr_labels = []
        function_group_labels = []
        finger_print_labels = []
        
        smiles_lst = []
        
        if FLAGS.get_frag:
            graph_labels = []
            # token_labels = []
            token_labels = torch.full((len(features), self.max_length), -1)
        
        
        batch = {}
        for i, ele in enumerate(features):
            ele = list(ele)
            lingua_inputs['smiles'].append(ele[0])
            lingua_inputs['smiles_len'].append(len(ele[0]['input_ids']))
            
            
            if FLAGS.get_frag:
                if isinstance(ele[1], list):
                    if self.align_batch or self.mask_batch:
                        ele[1] = ele[1][0] # original pair
                    else:
                        ele[1] = ele[1][1]
                if isinstance(ele[2], list):
                    if self.align_batch or self.mask_batch:
                        ele[2] = ele[2][0] # original pair
                    else:
                        ele[2] = ele[2][1]
                
            
            
            if not FLAGS.smiles_only:
                lingua_inputs['iupac'].append(ele[1])
                lingua_inputs['iupac_len'].append(len(ele[1]['input_ids']))
            
            graph_inputs.append(ele[2][0]) # omit the mask label
            gm_labels.append(np.array(ele[2][1]))
            
            if self.align_batch or self.mask_batch:
                ctr_labels.append(0) # all pos
            else:
                ctr_labels.append(ele[3]) # contains neg
            function_group_labels.append(ele[4])
            finger_print_labels.append(ele[5])
            
            if FLAGS.get_frag:
                # graph_labels.extend(ele[6])
                graph_labels.append(ele[6])
                # token_labels.append(ele[7])
                max_len = min(len(ele[7]), self.max_length-2)
                # ommit the cls and sep
                token_labels[i][1:max_len+1] = torch.tensor(ele[7][:max_len])

                smiles_lst.append(ele[8]) # smiles for debug
        
        # pad input
        if not FLAGS.smiles_only:
            lingua_inputs['iupac'] = self.tokenizer.pad(
                                        lingua_inputs['iupac'],
                                        padding="max_length",
                                        max_length=self.max_length,
                                        # truncation=True,
                                        return_tensors="pt",
                                    )
            
        lingua_inputs['smiles'] = self.smiles_tokenizer.pad(
                                    lingua_inputs['smiles'],
                                    padding="max_length",
                                    max_length=self.max_length,
                                    # truncation=True,
                                    return_tensors="pt",
                                )
        assert FLAGS.get_frag
        if self.mask_batch:
            # half smiles and graph mask token-level
            # half mask fragment-level conditional 

            # mask smiles:
            # firstly token-level random mask
            batch_size = len(ctr_labels)
            lingua_inputs['smiles']["mlm_input_ids"], lingua_inputs['smiles']["mlm_labels"] = self.mask_tokens(lingua_inputs['smiles']["input_ids"], self.smiles_tokenizer)
            # pick half to do the fragment-level mask
            h_batch_size = batch_size  // 2

            # random mask graph or smiles sample-wise at last h_batch_size sample
            # mask_smiles_lst = np.random.choice([True, False], h_batch_size, replace=True)

            mask_smiles_lst = np.zeros(h_batch_size).astype(np.bool)
            mask_smiles_lst[(h_batch_size//2) :] = True

            idx_lst = [t_id for t_id in range(h_batch_size, batch_size)]
            self.mask_tokens_group_simple(lingua_inputs['smiles'], token_labels, idx_lst, mask_smiles_lst)



            gm_masked_labels = []
            

            
            for idx, graph_ele in enumerate(graph_inputs):
                num_atoms = graph_ele.x.shape[0]
                
                all_indices = [i for i in range(num_atoms)]
                if idx < h_batch_size: # atom level mask
                    sample_num = max(int(len(all_indices) * self.mlm_probability), 1) # at least mask one atom
                    mask_indices = np.random.choice(all_indices, sample_num, replace=False)
                else: # fragment level mask
                    start_idx = idx - h_batch_size
                    mask_indices = []
                    if not mask_smiles_lst[start_idx]: # only mask graph
                        ele_graph_labels = graph_labels[idx]
                        group_num = max(ele_graph_labels) + 1
                        group_array = [i for i in range(group_num)]
                        mask_group_num = int(group_num * self.mlm_group_probability)
                        mask_group_num = max(1, mask_group_num) # for all labels are same(only one group), mask all! 
                        mask_group_array = random.sample(group_array, mask_group_num)
                        for j, ele_idx in enumerate(all_indices):
                            if ele_graph_labels[j] in mask_group_array:
                                mask_indices.append(ele_idx)
                    
                
                
                perm = mask_indices
                n_mask = len(perm)

                if n_mask:
                    mask_bond_idx = []
                    for i, ele in enumerate(graph_ele.edge_index.t()):
                        if ele[0].item() in perm or ele[1].item() in perm:
                            mask_bond_idx.append(i)

                
                    atom_mask_feature = torch.tensor([get_mask_atom_feature() for _ in range(n_mask)])
                    graph_ele.x[perm] = atom_mask_feature

                    if len(mask_bond_idx):
                        bond_mask_feature = torch.tensor([get_bond_mask_feature() for _ in range(len(mask_bond_idx))])
                        graph_ele.edge_attr[mask_bond_idx] = bond_mask_feature

                


                # for 2d
                gm_masked_labels_ele = np.array([-100 for _ in range(num_atoms)])
                if n_mask:
                    gm_masked_labels_ele[perm] = gm_labels[idx][perm]
                gm_masked_labels.append(torch.from_numpy(gm_masked_labels_ele))


            gm_masked_labels = collate_tokens(gm_masked_labels, pad_idx=-100, pad_to_multiple=8)
            batch['gm_masked_labels'] = gm_masked_labels


        # get graph attention
        graph_label_lst = []
        for ele in gm_labels:
            graph_label_lst.append(torch.tensor(ele))
        graph_label_pad_lst = collate_tokens(graph_label_lst, pad_idx=-100, pad_to_multiple=8)
        graph_attention_mask = graph_label_pad_lst.ne(-100)
        batch['graph_attention_mask'] = graph_attention_mask

        graph_inputs = Batch.from_data_list(graph_inputs)
        batch['graph'] = graph_inputs
        

        batch['contrastive_labels'] = torch.tensor(ctr_labels)
        if self.align_batch: # group align batch, no mask 
            batch['graph_labels'] = graph_labels
            batch['token_labels'] = token_labels
        
        batch['smiles_lst'] = smiles_lst

        self.switch_num += 1
        if self.switch_num % 3 == 0:
            self.mask_batch = True
            self.align_batch = False # positive pair, with mask
        elif self.switch_num % 3 == 1:
            self.mask_batch = False
            self.align_batch = True # positive pair, wo mask, group align
        else:
            self.mask_batch = False
            self.align_batch = False # pos + nega pair, wo mask, SGM
        
        if FLAGS.smiles_only:
            batch['lingua'] = lingua_inputs['smiles']
        else:
            batch['lingua'] = self._concat_lang(lingua_inputs, max_length=FLAGS.max_lingua_len, iupac_start_emb=self.smiles_tokenizer.vocab_size)
        # ctr labels, function group labels, fingerprint labels
        batch['function_group_labels'] = torch.tensor(function_group_labels)
        batch['fingerprint_labels'] = torch.tensor(finger_print_labels)
        # return batch
            

        return batch

            
        # simple group mask
    def mask_tokens_group_simple(self, inputs, group_labels, idx_lst, mask_smiles_lst):
        for i, idx in enumerate(idx_lst):
            if mask_smiles_lst[i]: # only mask smiles
                group_label = group_labels[idx]
                # group start from 0
                group_num = group_label.max().item() + 1
                group_array = [j for j in range(group_num)]
                mask_group_num = int(group_num * self.mlm_group_probability)
                mask_group_num = max(1, mask_group_num) # for all labels are same(only one group), mask all! 
                
                if not len(group_array): # empty, group num is 0
                    mask_group_array = []
                else:
                    mask_group_array = random.sample(group_array, mask_group_num)
                
                mlm_input_ids_row = inputs['input_ids'][idx].clone()
                mlm_labels = inputs['input_ids'][idx].clone()
                
                # order info
                mask_indices = torch.full(mlm_input_ids_row.shape, 0, dtype=torch.bool)
                for ele in mask_group_array:
                    mask_indices = mask_indices | (group_label == ele)
                
                mlm_labels[~mask_indices] = -100
                mlm_input_ids_row[mask_indices] = self.smiles_tokenizer.convert_tokens_to_ids(self.smiles_tokenizer.mask_token)
            else: # only mask graph
                mlm_input_ids_row = inputs['input_ids'][idx].clone()
                mlm_labels = inputs['input_ids'][idx].clone()
                mlm_labels[:] = -100

            inputs['mlm_input_ids'][idx] = mlm_input_ids_row
            inputs['mlm_labels'][idx] = mlm_labels

    # 
    def mask_tokens_group(self, inputs, group_labels, ctr_labels, mask_smiles=True):
        
        # inputs has been masked before
        assert 'input_ids' in inputs
        assert 'mlm_input_ids' in inputs
        assert 'mlm_labels' in inputs
        
        for i, ele in enumerate(ctr_labels):
            if ele == 0: # pair
                if not mask_smiles:
                    mlm_input_ids_row = inputs['input_ids'][i].clone()
                    mlm_labels = inputs['input_ids'][i].clone()
                    mlm_labels[:] = -100
                else:
                    group_label = group_labels[i]
                    # group start from 0
                    group_num = group_label.max().item() + 1
                    group_array = [i for i in range(group_num)]
                    mask_group_num = int(group_num * self.mlm_group_probability)
                    mask_group_num = max(1, mask_group_num) # for all labels are same(only one group), mask all! 
                    
                    if not len(group_array): # empty, group num is 0
                        mask_group_array = []
                    else:
                        mask_group_array = random.sample(group_array, mask_group_num)
                    
                    mlm_input_ids_row = inputs['input_ids'][i].clone()
                    mlm_labels = inputs['input_ids'][i].clone()
                    
                    # order info
                    mask_indices = torch.full(mlm_input_ids_row.shape, 0, dtype=torch.bool)
                    for ele in mask_group_array:
                        mask_indices = mask_indices | (group_label == ele)
                    
                    mlm_labels[~mask_indices] = -100
                    mlm_input_ids_row[mask_indices] = self.smiles_tokenizer.convert_tokens_to_ids(self.smiles_tokenizer.mask_token)
                    
                inputs['mlm_input_ids'][i] = mlm_input_ids_row
                inputs['mlm_labels'][i] = mlm_labels
        
    
    
    def mask_tokens(
        self, inputs: torch.Tensor, tokenizer, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def main(argv):
    torch.manual_seed(0)

    is_gpu = torch.cuda.is_available()
    
    FLAGS.num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    FLAGS.distributed = FLAGS.num_gpus > 1
    
    if FLAGS.local_rank == -1 or FLAGS.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not FLAGS.no_cuda else "cpu")
        
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(FLAGS.local_rank)
    #     device = torch.device("cuda", FLAGS.local_rank)
    #     torch.distributed.init_process_group(
    #         backend='nccl', init_method="env://"
    #     )
        
    


    config = RobertaConfig(
        vocab_size=FLAGS.vocab_size,
        max_position_embeddings=FLAGS.max_position_embeddings,
        # num_attention_heads=FLAGS.num_attention_heads,
        # num_hidden_layers=FLAGS.num_hidden_layers,
        type_vocab_size=FLAGS.type_vocab_size,
        contrastive_class_num=2,
        pooler_type=FLAGS.pooler_type
    ) # parameter size: ~88w


    # if FLAGS.tokenizer_path:
    #     tokenizer_path = FLAGS.tokenizer_path
    # elif FLAGS.tokenizer_type.upper() == "BPE":
    #     tokenizer_path = FLAGS.output_tokenizer_dir
    #     if not os.path.isdir(tokenizer_path):
    #         os.makedirs(tokenizer_path)

    #     tokenizer = ByteLevelBPETokenizer()
    #     print('Training tokenizer...')
    #     tokenizer.train(files=FLAGS.dataset_path, vocab_size=FLAGS.vocab_size, min_frequency=FLAGS.BPE_min_frequency, special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])
    #     tokenizer.save_model(tokenizer_path)
    #     print('Tokenizer saved')
    # else:
    #     print("Please provide a tokenizer path if using the SMILES tokenizer")

    # tokenizer = RobertaTokenizerFast.from_pretrained(FLAGS.tokenizer_path, max_len=FLAGS.max_tokenizer_len,truncate=True)
    # vocab_path = "Datas/tokenizer_dir/iupac_reg/new_vocab.json"
 

    tokenizer = IUPACTokenizer.from_pretrained(FLAGS.tokenizer_path)
    print('Trained tokenizer iupac vocab_size: ', tokenizer.vocab_size)
    smiles_tokenizer = SmilesTokenizer.from_pretrained(FLAGS.smiles_tokenizer_path)
    print('Trained smiles tokenizer iupac vocab_size: ', smiles_tokenizer.vocab_size)
   

    # todo should decorate the reoberta with LM head
    config.vocab_size = tokenizer.vocab_size
    
    # construct gnn config
    gnn_config = {
        "gnn_number_layer": FLAGS.gnn_number_layer,
        "gnn_dropout": FLAGS.gnn_dropout,
        "conv_encode_edge": FLAGS.conv_encode_edge,
        "gnn_embed_dim": FLAGS.gnn_embed_dim,
        "gnn_aggr": FLAGS.gnn_aggr,
        "gnn_norm": FLAGS.gnn_norm,
        "gnn_act": FLAGS.gnn_act,
        "atom_vocab_size": FLAGS.atom_vocab_size,
        "graph_max_seq_size": FLAGS.graph_max_seq_size,
    }
    
        
    # smiles_config = RobertaConfig(
    #     vocab_size=smiles_tokenizer.vocab_size,
    #     max_position_embeddings=FLAGS.max_position_embeddings,
    #     # num_attention_heads=FLAGS.num_attention_heads,
    #     # num_hidden_layers=FLAGS.num_hidden_layers,
    #     type_vocab_size=FLAGS.type_vocab_size,
    #     temp=FLAGS.temp,
    #     pooler_type=FLAGS.pooler_type
    # )
    if FLAGS.smiles_only:
        config.vocab_size = smiles_tokenizer.vocab_size
        config.contrastive_class_num = 2
    else:
        config.vocab_size = tokenizer.vocab_size + smiles_tokenizer.vocab_size
        config.contrastive_class_num = 3

    model = MultilingualModelUNI(config, gnn_config, atom_vocab_size=FLAGS.atom_vocab_size, check_frag=FLAGS.check_frag)

    
    # model.lang_roberta.resize_token_embeddings(len(tokenizer))
  
    print('Training dual reberta model...')
    # model = RobertaForMaskedLM(config=config)
    print(f"Model size: {model.num_parameters()} parameters.")
    # wandb.login()
    # dataset = RawTextDataset(tokenizer=tokenizer, file_path=FLAGS.dataset_path, block_size=FLAGS.tokenizer_block_size)
    # dataset_smiles = RawTextDataset(tokenizer=tokenizer_smiles, file_path=FLAGS.dataset_path_smiles, block_size=FLAGS.tokenizer_block_size_smiles)

    

    #function_group_file: str, finger_print_file: str, replace_prob: float = 0.5, replace_graph_prob: float = 0.5, replace_iupac_prob: int = 0)
    dataset_all = SmilesIUPACDatasetDualSplitUNI(tokenizer_iupac=tokenizer, tokenizer_smiles=smiles_tokenizer, \
            file_path=FLAGS.dataset_path, block_size=FLAGS.tokenizer_block_size, atom_vocab_file=FLAGS.atom_vocab_file, \
            function_group_file = FLAGS.function_group_file, finger_print_file=FLAGS.finger_print_file, replace_prob=FLAGS.replace_prob, replace_graph_prob=FLAGS.replace_graph_prob, replace_iupac_prob=FLAGS.replace_iupac_prob, get_frag=FLAGS.get_frag, smiles_only=FLAGS.smiles_only, detect_fg=FLAGS.detect_fg,
            mix_brics=FLAGS.mix_brics, brics_plus=FLAGS.brics_plus
            )
    
    
    train_size = max(int(FLAGS.frac_train * len(dataset_all)), 1)
    eval_size = len(dataset_all) - train_size
    print(f"Train size: {train_size}")
    print(f"Eval size: {eval_size}")

    train_dataset, eval_dataset = random_split(dataset_all, [train_size, eval_size])
    print('mlm prob: {}'.format(FLAGS.mlm_probability))
    data_collator = OurDataCollatorWithPadding(
        tokenizer=tokenizer,  smiles_tokenizer=smiles_tokenizer, mlm=True, mlm_probability=FLAGS.mlm_probability, mlm_group_probability=FLAGS.mlm_group_probability, max_length=FLAGS.max_tokenizer_len
    )

    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        save_strategy='epoch',
        # logging_strategy='epoch',
        # eval_steps=FLAGS.eval_steps,
        load_best_model_at_end=True,
        logging_steps=FLAGS.logging_steps,
        output_dir=os.path.join(FLAGS.output_dir, FLAGS.run_name),
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        num_train_epochs=FLAGS.num_train_epochs,
        per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        per_device_eval_batch_size=FLAGS.per_device_eval_batch_size,
        # save_steps=FLAGS.save_steps,
        save_total_limit=FLAGS.save_total_limit,
        fp16 = is_gpu and FLAGS.fp16, # fp16 only works on CUDA devices
        report_to="tensorboard",
        run_name=FLAGS.run_name,
        dataloader_drop_last=True,
    )
    # training_args.logging_steps = 1
    training_args.logging_first_step = True

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    if FLAGS.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model(os.path.join(FLAGS.output_dir, FLAGS.run_name, "final"))


if __name__ == '__main__':
    app.run(main)
