from cProfile import label
import os
import torch
from rdkit.Chem.PandasTools import LoadSDF
from pe_2d.utils_pe_seq import InputExample, convert_examples_seq_to_features
import numpy as np
from utils.mol import smiles2graph



def convert_to_smiles_seq_examples(smiles_ids):
    input_examples = []
    for smiles_id in smiles_ids:
        input_examples.append(InputExample(
                seq=smiles_id,
            ))
    return input_examples


class FinetuneDDIDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, pair_ids, tokenizer, smiles_dict, labels, data_suffix=".sdf", get_labels=True):
        # collect smiles
        self.smiles_lst = []
        self.pair_labels = []
        for sdf_ids in pair_ids:
            smiles_pair = []
            for sdf_id in sdf_ids:
                smiles_pair.append(smiles_dict[sdf_id])
                # smiles_pair.append(smiles)
            self.smiles_lst.append(smiles_pair)
        
        
        
        # self.pair_labels = None
        # self.labels = None
        # if labels is not None:
        
        self.get_labels = get_labels
        
        self.pair_labels = labels
        assert len(self.pair_labels) == len(self.smiles_lst)
        self.labels = np.array(self.pair_labels)
        
        
        np_smiles_lst = np.array(self.smiles_lst)
        left_smiles_lst = np_smiles_lst[:, 0].tolist()
        right_smiles_lst = np_smiles_lst[:, 1].tolist()
        
        
        
        # tokenizer
        left_input_examples = convert_to_smiles_seq_examples(left_smiles_lst)
        right_input_examples = convert_to_smiles_seq_examples(right_smiles_lst)
        self.left_encodings = convert_examples_seq_to_features(left_input_examples, max_seq_length=128,tokenizer=tokenizer)
        self.right_encodings = convert_examples_seq_to_features(right_input_examples, max_seq_length=128,tokenizer=tokenizer)
        
        
    def __len__(self):
        return len(self.smiles_lst)
    
    
    def __getitem__(self, idx):
        item = {}
        
        # no need to concat
        
        item['left_input_ids']=self.left_encodings[idx].input_ids
        item['left_attention_mask']=self.left_encodings[idx].attention_mask
    
        item['right_input_ids'] = self.right_encodings[idx].input_ids
        item['right_attention_mask'] = self.right_encodings[idx].attention_mask
        
        smiles_pair = self.smiles_lst[idx]
        
        left_graph, _ = smiles2graph(smiles_pair[0])
        right_graph, _ = smiles2graph(smiles_pair[1])
        
        
        # graph
        item['left_graph'] = left_graph
        item['right_graph'] = right_graph
        
        
        # label
        # if self.pair_labels is not None:
        if self.get_labels:
            item["labels"] = torch.tensor(self.pair_labels[idx], dtype=torch.float)
        
        return item
    
    # def __init__(self, df, tokenizer, include_labels=True, use_struct_pos=True, tasks_wanted=None, iupac_only=False, lang_only=False, gnn_only=False, iupac_smiles_concat=False, graph_uni=False, use_rdkit_feature=False):
    #     #df = df[(df['iupac_ids'] != 'Request error') &  (df['iupac_ids'] != '')] # assume df has iupac field
    #     df = df[(df['iupac_ids'] != 'Request error') &  (df['iupac_ids'] != '') & (pd.isna(df['iupac_ids']) == False)] # assume df has iupac field
    #     self.iupac_only = iupac_only
    #     self.lang_only = lang_only
    #     self.gnn_only = gnn_only
        
    #     self.use_struct_pos = use_struct_pos
        
    #     # concat smiles and iupac as one sequence
    #     self.iupac_smiles_concat = iupac_smiles_concat
    #     self.graph_uni = graph_uni
    #     # self.iupac_ids = False
    #     # if 'iupac_ids' in df.keys():
            
    #         # filter
        
    #     self.fp_features = None
    #     if use_rdkit_feature:
    #         self.fp_features = []
    #         pool = multiprocessing.Pool(24)
    #         smiles_lst = df["smile_ids"].tolist()
    #         total = len(smiles_lst)
            
    #         for res in tqdm(pool.imap(rdkit_2d_features_normalized_generator, smiles_lst, chunksize=10), total=total):
    #             replace_token = 0
    #             fp_feature = np.where(np.isnan(res), replace_token, res)
    #             self.fp_features.append(np.float32(fp_feature))    
        
        
    #     # for smile in df["smile_ids"].tolist():
    #     #     fp_feature = rdkit_2d_features_normalized_generator(smile)
    #     #     replace_token = 0
    #     #     fp_feature = np.where(np.isnan(fp_feature), replace_token, fp_feature)
    #     #     self.fp_features.append(np.float32(fp_feature))
        
        
    #     if self.lang_only:
    #         self.smiles_lst = df["smile_ids"].tolist() # for uni
    #         if self.iupac_smiles_concat:
    #             input_examples = convert_to_iupac_seq_examples(df["iupac_ids"].tolist())
    #             self.iupac_features, self.iupac_length = convert_examples_seq_to_features_wlen(input_examples, max_seq_length=128,tokenizer=tokenizer[0]) 
    #             input_examples = convert_to_smiles_seq_examples(df["smile_ids"].tolist())
    #             self.smiles_features, self.smiles_length = convert_examples_seq_to_features_wlen(input_examples, max_seq_length=128,tokenizer=tokenizer[1])

    #             self.iupac_start_emb = tokenizer[1].vocab_size # smiles token size of the begining

    #             self.labels = df.iloc[:, 2].values

    #         elif self.iupac_only:
    #             if use_struct_pos:
    #                 input_examples = convert_to_input_examples(df["iupac_ids"].tolist())
    #                 self.features = convert_examples_with_strucpos_to_features(input_examples, max_seq_length=128,
    #                             max_pos_depth=16, tokenizer=tokenizer[0])
    #             else:
    #                 input_examples = convert_to_iupac_seq_examples(df["iupac_ids"].tolist())
    #                 self.features = convert_examples_seq_to_features(input_examples, max_seq_length=128,tokenizer=tokenizer[0]) 
    #                 # self.encodings = tokenizer(df["smiles"].tolist(), truncation=True, padding=True)
    #             self.iupac_ids = True
    #             self.labels = df.iloc[:, 2].values
    #         else:
    #             input_examples = convert_to_smiles_seq_examples(df["smile_ids"].tolist())
    #             self.encodings = convert_examples_seq_to_features(input_examples, max_seq_length=128,tokenizer=tokenizer[1])
    #             self.labels = df.iloc[:, 1].values
    #     elif self.gnn_only:
    #         # save smiles_ids
    #         self.smiles_lst = df["smile_ids"].tolist()
    #     else:
    #         raise NotImplementedError
    
        
    #     if tasks_wanted is not None:
    #         if len(tasks_wanted) == 1:
    #             self.labels = df[tasks_wanted[0]].values
    #         else:
    #             labels = []
    #             for task in tasks_wanted:
    #                 labels.append(df[task].values.reshape(-1, 1))
    #             self.labels = np.concatenate(labels, axis=1)
            
    #     task_weight_cols = list(df.columns[1:-2]) # X ,..., smiles_ids, iupac_ids
    #     self.task_weights = None
    #     if 'w' in task_weight_cols or 'w1' in task_weight_cols:
    #         # import pdb; pdb.set_trace()
    #         task_weights = []
    #         for task in tasks_wanted:
    #             task_idx = task_weight_cols.index(task)
    #             task_weight_col_name = 'w' + str(task_idx + 1)
    #             if len(task_weight_cols) == 2:
    #                 task_weight_col_name = task_weight_cols[1]
    #             task_weights.append(df[task_weight_col_name].tolist())
            
    #         task_weights = np.array(task_weights, dtype=np.float32)
            
    #         self.task_weights = task_weights.T
    #     self.include_labels = include_labels


    # def _concat_lang(self, smiles_inputs, iupac_inputs, smile_len, iupac_len, concat_max_length=256):
    #     return_dict = {}
    #     # import pdb;pdb.set_trace()
    #     return_dict['input_ids'] = torch.full((1, concat_max_length), 1)[0] # padding: 1
    #     return_dict['attention_mask'] = torch.full((1, concat_max_length), 0)[0] # attention mask, default: 0
        

    #     return_dict['input_ids'][:smile_len] = torch.tensor(smiles_inputs.input_ids[:smile_len])
        
    #     iupac_input_ids = torch.tensor(iupac_inputs.input_ids[1:iupac_len]) # earse the iupac cls token
    #     iupac_input_ids[:-1] += self.iupac_start_emb # except the sep token
    #     return_dict['input_ids'][smile_len: smile_len + iupac_len - 1] = iupac_input_ids
        
    #     return_dict['attention_mask'][:smile_len] = torch.tensor(smiles_inputs.attention_mask[:smile_len])
    #     return_dict['attention_mask'][smile_len: smile_len + iupac_len - 1] = torch.tensor(iupac_inputs.attention_mask[1:iupac_len]) # erase the iupac cls token    
    #     return return_dict


    # def __getitem__(self, idx):
    #     # get smiles and transfer to graph
    #     if self.gnn_only:
    #         item = {}
    #         smiles = self.smiles_lst[idx]
    #         graph, _ = smiles2graph(smiles)
    #         item['graph'] = graph
    #     elif self.lang_only:
    #         if self.iupac_smiles_concat:
    #             item = {}
    #             # concat smiles and iupac features
    #             item = self._concat_lang(self.smiles_features[idx], self.iupac_features[idx], \
    #                 self.smiles_length[idx], self.iupac_length[idx], FLAGS.max_concat_len)
    #         elif self.iupac_only:
    #             if self.use_struct_pos:
    #                 item = {}
    #                 item['input_ids']=self.features[idx].input_ids
    #                 item['attention_mask']=self.features[idx].attention_mask
    #                 item['strucpos_ids']=self.features[idx].strucpos_ids
    #             else:
    #                 #item = {key: torch.tensor(val[idx]) for key, val in self.features.items()}
    #                 item = {}
    #                 item['input_ids']=self.features[idx].input_ids
    #                 item['attention_mask']=self.features[idx].attention_mask
    #         else:
    #             item = {}
    #             item['input_ids']=self.encodings[idx].input_ids
    #             item['attention_mask']=self.encodings[idx].attention_mask
            
    #         if self.graph_uni:
    #             smiles = self.smiles_lst[idx]
    #             graph, _ = smiles2graph(smiles)
    #             item['graph'] = graph # add graph 
    #     else:
    #         raise NotImplementedError
        
    #     if self.include_labels and self.labels is not None:
    #         item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
    #         if self.task_weights is not None:
    #             item['weight'] = torch.tensor(self.task_weights[idx], dtype=torch.float)
        
    #     if self.fp_features is not None:
    #         item['fp_feature'] = self.fp_features[idx] 
        
    #     return item

    # def __len__(self):
    #     if self.gnn_only:
    #         return len(self.smiles_lst)
    #     if self.iupac_smiles_concat:
    #         return len(self.iupac_features)

    #     if self.iupac_only:
    #         return len(self.features)#["input_ids"])
    #     return len(self.encodings)#["input_ids"])
