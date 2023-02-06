import os
from torch import nn
import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset
from pe_2d.utils_pe_seq import InputExample, convert_examples_seq_to_features
from utils.mol import smiles2graph
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from utils.multilingual_regression import RobertaFeatureHead, RobertaHead
from gcn import DeeperGCN
from models import Pooler
from utils.raw_text_dataset import collate_tokens
from torch.nn import SmoothL1Loss
# from .molecule_datasets import mol_to_graph_data_obj_simple

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


def convert_to_smiles_seq_examples(smiles_ids):
    input_examples = []
    for smiles_id in smiles_ids:
        input_examples.append(InputExample(
                seq=smiles_id,
            ))
    return input_examples


class MoleculeProteinDataset(InMemoryDataset):
    def __init__(self, root, dataset, smiles_tokenizer, mode, include_labels=False):
        super(InMemoryDataset, self).__init__()
        self.root = root
        self.dataset = dataset
        datapath = os.path.join(self.root, self.dataset, '{}.csv'.format(mode))
        print('datapath\t', datapath)
        
        self.smiles_tokenizer = smiles_tokenizer

        self.process_molecule()
        self.process_protein()

        df = pd.read_csv(datapath)
        self.molecule_index_list = df['smiles_id'].tolist()
        self.protein_index_list = df['target_id'].tolist()
        self.label_list = df['affinity'].tolist()
        self.label_list = torch.FloatTensor(self.label_list)
        self.labels = self.label_list
        self.include_labels = include_labels

        return
    
    def process_molecule(self):
        input_path = os.path.join(self.root, self.dataset, 'smiles.csv')
        input_df = pd.read_csv(input_path, sep=',')
        self.smiles_list = input_df['smiles']
        input_examples = convert_to_smiles_seq_examples(self.smiles_list)
        self.encodings = convert_examples_seq_to_features(input_examples, max_seq_length=128,tokenizer=self.smiles_tokenizer)


    # def process_molecule(self):
    #     input_path = os.path.join(self.root, self.dataset, 'smiles.csv')
    #     input_df = pd.read_csv(input_path, sep=',')
    #     smiles_list = input_df['smiles']

    #     rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #     preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in rdkit_mol_objs_list]
    #     preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in preprocessed_rdkit_mol_objs_list]
    #     assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    #     assert len(smiles_list) == len(preprocessed_smiles_list)

    #     smiles_list, rdkit_mol_objs = preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list

    #     data_list = []
    #     for i in range(len(smiles_list)):
    #         rdkit_mol = rdkit_mol_objs[i]
    #         if rdkit_mol != None:
    #             data = mol_to_graph_data_obj_simple(rdkit_mol)
    #             data.id = torch.tensor([i])
    #             data_list.append(data)

    #     self.molecule_list = data_list
    #     return

    def process_protein(self):
        datapath = os.path.join(self.root, self.dataset, 'protein.csv')

        input_df = pd.read_csv(datapath, sep=',')
        protein_list = input_df['protein'].tolist()

        self.protein_list = [seq_cat(t) for t in protein_list]
        self.protein_list = torch.LongTensor(self.protein_list)
        return

    def __getitem__(self, idx):
        item = {}
        # molecule = self.molecule_list[self.molecule_index_list[idx]]
        item['input_ids']=self.encodings[self.molecule_index_list[idx]].input_ids
        item['attention_mask']=self.encodings[self.molecule_index_list[idx]].attention_mask
        smiles = self.smiles_list[self.molecule_index_list[idx]]
        graph, _ = smiles2graph(smiles)
        item['graph'] = graph # add graph 
        protein = self.protein_list[self.protein_index_list[idx]]
        item['protein_encoding'] = protein
        if self.include_labels:
            label = self.label_list[idx]
            item['label'] = label
        return item

    def __len__(self):
        return len(self.label_list)


class ProteinModel(nn.Module):
    def __init__(self, emb_dim=128, num_features=25, output_dim=128, n_filters=32, kernel_size=8):
        super(ProteinModel, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.intermediate_dim = emb_dim - kernel_size + 1

        self.embedding = nn.Embedding(num_features+1, emb_dim)
        self.n_filters = n_filters
        self.conv1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=kernel_size)
        self.fc = nn.Linear(n_filters*self.intermediate_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = x.view(-1, self.n_filters*self.intermediate_dim)
        x = self.fc(x)
        return x

class MoleculeProteinModel(nn.Module):
    def __init__(self, molecule_model, protein_model, molecule_emb_dim, protein_emb_dim, output_dim=1, dropout=0.2):
        super(MoleculeProteinModel, self).__init__()
        self.fc1 = nn.Linear(molecule_emb_dim+protein_emb_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, output_dim)
        self.molecule_model = molecule_model
        self.protein_model = protein_model
        self.pool = global_mean_pool
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, molecule, protein):
        molecule_node_representation = self.molecule_model(molecule)
        molecule_representation = self.pool(molecule_node_representation, molecule.batch)
        protein_representation = self.protein_model(protein)

        x = torch.cat([molecule_representation, protein_representation], dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x


class MultilingualModalUNIDTI(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config, gcn_config, is_regression=False, use_label_weight=False, use_rdkit_feature=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_tasks = config.num_tasks # sider have 27 binary tasks, maybe multi head is useful for multi label classification

        self.register_buffer("norm_mean", torch.tensor(config.norm_mean))
            # Replace any 0 stddev norms with 1
        self.register_buffer(
            "norm_std",
            torch.tensor(
                [label_std if label_std != 0 else 1 for label_std in config.norm_std]
            ),
        )
        
        if self.num_tasks > 1:
            assert self.num_labels == 2 # binary multi label classification

        # iupac and smiles has same 
        from multimodal.modeling_roberta import RobertaModel
        self.lang_roberta = RobertaModel(config, add_pooling_layer=True)
        # self.smiles_roberta = RobertaModel(smiles_config, add_pooling_layer=True)
        
        self.lang_pooler = Pooler(config.pooler_type)
        self.gnn = DeeperGCN(gcn_config)
        
        self.gcn_config = gcn_config
        self.config = config
        
        # transfer from gcn embeddings to lang shape
        self.gcn_embedding = nn.Linear(gcn_config['gnn_embed_dim'], config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        
        
        self.use_rdkit_feature = use_rdkit_feature
        self.use_label_weight = use_label_weight

        
        if self.use_rdkit_feature:
            self.head = RobertaFeatureHead(config, regression=is_regression)
        else:
            self.head = RobertaHead(config, regression=is_regression)
        
        # self.head = RobertaFeatureHead(config, regression=is_regression)

        
        self.is_regression = is_regression
        if is_regression:
            self.register_buffer("norm_mean", torch.tensor(config.norm_mean))
            # Replace any 0 stddev norms with 1
            self.register_buffer(
                "norm_std",
                torch.tensor(
                    [label_std if label_std != 0 else 1 for label_std in config.norm_std]
                ),
            )

        self.init_weights()
        
        self.task_weight = None

        self.protein = ProteinModel(emb_dim=config.hidden_size, output_dim=config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size + config.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        # self.molecule_model = molecule_model
        # self.protein_model = protein_model
        # self.pool = global_mean_pool
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    
    def set_task_weight(self, task_weight):
        self.task_weight = task_weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        
        graph=None,
        # strucpos_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        weight=None,
        output_attentions=None,
        output_hidden_states=None,
        fp_feature = None,
        return_dict=None,
        protein_encoding=None,
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # graph_inputs = Batch.from_data_list(graph)
        graph.to(self.device)
        gcn_output = self.gnn(graph)
        # concat graph atome embeddings and langua embeddings
        gcn_embedding_output = self.gcn_embedding(gcn_output[1])
        gcn_embedding_output = self.LayerNorm(gcn_embedding_output)
        gcn_embedding_output = self.dropout(gcn_embedding_output)


        # pad the gcn_embedding same shape with pos_coord_matrix_pad
        gcn_embedding_lst = []
        batch_size = input_ids.shape[0]
        batch_idx = graph.batch
        
        graph_attention_mask = []
        for bs in range(batch_size):
            gcn_embedding_lst.append(gcn_embedding_output[batch_idx == bs])
            atom_num = (batch_idx == bs).sum().item()
            graph_attention_mask.append(torch.tensor([1 for _ in range(atom_num)]).to(self.device))
        
        graph_attention_mask = collate_tokens(graph_attention_mask, pad_idx=0, pad_to_multiple=8)
        graph_attention_mask = graph_attention_mask.to(torch.bool)

        lang_gcn_outputs, lang_gcn_attention_mask = self.lang_roberta(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=lingua['token_type_ids'],
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True if self.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,

            graph_input = gcn_embedding_lst,
            graph_batch = graph.batch,
            # graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
            gnn_mask_labels = None,
            graph_attention_mask = graph_attention_mask,
            )
        lang_gcn_pooler_output = self.lang_pooler(lang_gcn_attention_mask, lang_gcn_outputs)
        
        protein_representation = self.protein(protein_encoding) # get protein representations

        x = torch.cat([lang_gcn_pooler_output, protein_representation], dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        loss_fct = SmoothL1Loss()
        if labels is None:
            return self.unnormalize_logits(x).float()
        normalized_labels = self.normalize_logits(labels).float()
        loss = loss_fct(x.view(-1), normalized_labels)

        return [loss]


    def normalize_logits(self, tensor):
        return (tensor - self.norm_mean) / self.norm_std

    def unnormalize_logits(self, tensor):
        return (tensor * self.norm_std) + self.norm_mean