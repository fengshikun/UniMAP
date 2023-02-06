from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from gcn import DeeperGCN
from models import Pooler
from torch_geometric.data import Batch
import torch.nn.functional as F
from utils.raw_text_dataset import collate_tokens
import copy
from transformers.models.t5.modeling_t5 import T5Stack, T5PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers import T5Config

class GNNHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # self.activation_fn = nn.ReLU()
        self.activation_fn = nn.GELU()
        # self.norm = nn.BatchNorm1d(config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        num_tasks = config.num_tasks

        if num_tasks > 1:
            
 
            # output_dim = config.num_labels
            # self.out_weight = nn.Parameter(torch.Tensor(num_tasks, output_dim, config.hidden_size))
            # self.out_proj = nn.ModuleList([nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), 
            #     nn.Dropout(config.hidden_dropout_prob),
            #     nn.ReLU(), nn.Linear(config.hidden_size, output_dim))
            #     for _ in range(num_tasks)])

            output_dim = num_tasks
            self.out_proj = nn.Linear(config.hidden_size, output_dim)

        else:
            output_dim = config.num_labels
            self.out_proj = nn.Linear(config.hidden_size, output_dim)

    def forward(self, features, only_feat=False):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        # if x.shape[0] == 1:
        #     x = torch.cat([x, x], dim=0)
        #     x = self.norm(x)
        #     x = x[0]
        # else:
        #     x = self.norm(x)
        x = self.norm(x)

        
            
        
        if only_feat:
            return x
        
        if isinstance(self.out_proj, nn.ModuleList):
            output = []
            # ex = x / torch.norm(x, 2, 1, keepdim=True)
            for i, proj in enumerate(self.out_proj):
                # weight = self.out_weight[i]
                # ew = weight / torch.norm(weight, 2, 1, keepdim=True)
                # cos = torch.mm(ex, ew.t())
                # output.append(cos * 64)
                

                output.append(proj(x))
            return output
        else:
            x = self.out_proj(x)
            return x

class RobertaHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, regression=False, dual_input=False):
        super().__init__()
        if dual_input:
            self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        num_tasks = config.num_tasks
        if num_tasks > 1:
            output_dim = num_tasks
        else:
            # output_dim = 1 # for bce loss
            output_dim = config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, output_dim)
        self.regression = regression

        if not self.regression:
            self.norm = nn.LayerNorm(config.hidden_size)
            self.gelu = nn.GELU()
            # self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
            # self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, features, only_feat=False):
        x = self.dropout(features)
        x = self.dense(x)
        if self.regression:
            x = torch.relu(x)
        else:
            x = self.gelu(x)
            # x = torch.tanh(x)
        # try diff norm: batch norm, layernorm        
        if only_feat:
            return x
        x = self.dropout(x)
        if not self.regression:
            x = self.norm(x)

        # x = self.dense2(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.norm2(x)

        x = self.out_proj(x)
        return x


class MultiModalHead(nn.Module):
    def __init__(self, config, regression=False):
        super().__init__()
        self.gnn_head = GNNHead(config)
        self.lang_head = RobertaHead(config, regression=regression)
        
        
        num_tasks = config.num_tasks
        if num_tasks > 1:
            output_dim = num_tasks
        else:
            output_dim = config.num_labels
        self.out_proj = nn.Linear(config.hidden_size * 2, output_dim)

    def forward(self, x0, x1):
        x0 = self.gnn_head(x0, True)
        x1 = self.lang_head(x0, True)
        if len(x0.shape) == 1:
            x0 = x0.reshape(1, -1)
            x1 = x1.reshape(1, -1)
        try:
            x = torch.cat([x0, x1], dim=1)
        except:
            print("debug")
            import pdb; pdb.set_trace()
        x = self.out_proj(x)
        return x

  
class MultilingualModal(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config, gcn_config, is_regression=False, gnn_only=False, lang_only=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_tasks = config.num_tasks # sider have 27 binary tasks, maybe multi head is useful for multi label classification
        
        if self.num_tasks > 1:
            assert self.num_labels == 2 # binary multi label classification

        # iupac and smiles has same 
        self.lang_roberta = RobertaModel(config, add_pooling_layer=True)
        self.lang_pooler = Pooler(config.pooler_type)
        self.gnn = DeeperGCN(gcn_config)
        
        self.gnn_only = gnn_only
        self.lang_only = lang_only
        if gnn_only:
            self.head = GNNHead(config)
        elif lang_only:
            self.head = RobertaHead(config, regression=is_regression)
        else:
            self.head = MultiModalHead(config, regression=is_regression)
        
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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
        graph_emb = gcn_output[0]
        if self.gnn_only:
            logits = self.head(graph_emb)
        else:
            outputs = self.lang_roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            lang_pooler_output = self.lang_pooler(attention_mask, outputs)
            if self.lang_only:
                # import pdb; pdb.set_trace()
                logits = self.head(lang_pooler_output)
            else:
                logits = self.head(graph_emb, lang_pooler_output)

        # sequence_output = (
        #     outputs.last_hidden_state,
        #     outputs_smiles.last_hidden_state,
            
        # )
        # # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)

        if labels is None:
            if self.is_regression:
                logits =  self.unnormalize_logits(logits).float()
            if len(logits) == 1:
                logits = logits.reshape(1, -1)
            return logits

        loss = None
        if labels is not None:
            if not self.is_regression:
                if self.num_tasks > 1:
                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                    loss = loss_fct(logits, labels)
                    loss = loss.mean()
                    # import pdb; pdb.set_trace()
                    
                    
                    # loss_fct = CrossEntropyLoss()
                    # loss = 0
                    # for i, l_ele in enumerate(logits):
                    #     loss += loss_fct(l_ele.view(-1, self.num_labels), labels[:,i:i+1].long().view(-1))
                    #     # import pdb; pdb.set_trace()
                    # loss = loss / self.num_tasks
                    # import pdb; pdb.set_trace()
        
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels.long().view(-1)
                    )
            else: # regression
                normalized_labels = self.normalize_logits(labels).float()
                # loss_fct = MSELoss() # todo smooth L1
                loss_fct = SmoothL1Loss()
                loss = loss_fct(logits.view(-1), normalized_labels.view(-1))

        
        
        return [loss]

    def normalize_logits(self, tensor):
        return (tensor - self.norm_mean) / self.norm_std

    def unnormalize_logits(self, tensor):
        return (tensor * self.norm_std) + self.norm_mean



class MultilingualModalSplit(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config, smiles_config, gcn_config, is_regression=False, gnn_only=False, lang_only=False, \
        iupac_only=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_tasks = config.num_tasks # sider have 27 binary tasks, maybe multi head is useful for multi label classification
        
        if self.num_tasks > 1:
            assert self.num_labels == 2 # binary multi label classification

        # iupac and smiles has same 
        self.iupac_roberta = RobertaModel(config, add_pooling_layer=True)
        self.smiles_roberta = RobertaModel(smiles_config, add_pooling_layer=True)
        
        self.lang_pooler = Pooler(config.pooler_type)
        self.gnn = DeeperGCN(gcn_config)
        
        self.gnn_only = gnn_only
        self.lang_only = lang_only
        self.iupac_only = iupac_only
        if gnn_only:
            self.head = GNNHead(config)
        elif lang_only:
            self.head = RobertaHead(config, regression=is_regression)
        else:
            # todo not
            raise NotImplementedError
            self.head = MultiModalHead(config, regression=is_regression)
        
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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
        
        
        # todo implement lang(iupac) and smiles aggregate code
        if self.gnn_only:
            graph.to(self.device)
            gcn_output = self.gnn(graph)
            graph_emb = gcn_output[0]
            logits = self.head(graph_emb)
        elif self.lang_only:
            if self.iupac_only:
                infer_model = self.iupac_roberta
            else:
                infer_model = self.smiles_roberta
                
            outputs = infer_model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            lang_pooler_output = self.lang_pooler(attention_mask, outputs)
            logits = self.head(lang_pooler_output)        
        else:
            raise NotImplementedError
            logits = self.head(graph_emb, lang_pooler_output)

        # sequence_output = (
        #     outputs.last_hidden_state,
        #     outputs_smiles.last_hidden_state,
            
        # )
        # # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)

        if labels is None:
            if self.is_regression:
                logits =  self.unnormalize_logits(logits).float()
            if len(logits) == 1:
                logits = logits.reshape(1, -1)
            return logits

        loss = None
        if labels is not None:
            if not self.is_regression:
                if self.num_tasks > 1:
                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                    loss = loss_fct(logits, labels)
                    loss = loss.mean()
                    # import pdb; pdb.set_trace()
                    
                    
                    # loss_fct = CrossEntropyLoss()
                    # loss = 0
                    # for i, l_ele in enumerate(logits):
                    #     loss += loss_fct(l_ele.view(-1, self.num_labels), labels[:,i:i+1].long().view(-1))
                    #     # import pdb; pdb.set_trace()
                    # loss = loss / self.num_tasks
                    # import pdb; pdb.set_trace()
        
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels.long().view(-1)
                    )
            else: # regression
                normalized_labels = self.normalize_logits(labels).float()
                # loss_fct = MSELoss() # todo smooth L1
                loss_fct = SmoothL1Loss()
                loss = loss_fct(logits.view(-1), normalized_labels.view(-1))

        
        
        return [loss]

    def normalize_logits(self, tensor):
        return (tensor - self.norm_mean) / self.norm_std

    def unnormalize_logits(self, tensor):
        return (tensor * self.norm_std) + self.norm_mean



class RobertaFeatureHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, regression=False, fp_dim=200, dual_input=False):
        super().__init__()
        if dual_input:
            self.dense = nn.Linear((config.hidden_size + fp_dim) * 2, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size + fp_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        num_tasks = config.num_tasks
        if num_tasks > 1:
            output_dim = num_tasks
        else:
            # output_dim = 1 # for bce loss
            output_dim = config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, output_dim)
        self.regression = regression

        if not self.regression:	
            self.norm = nn.LayerNorm(config.hidden_size)	
            self.gelu = nn.GELU()

    def forward(self, features, fp_features, only_feat=False):
        x = self.dropout(features)
        x = torch.cat((x, fp_features), dim=1)
        x = self.dense(x)
        if self.regression:
            x = torch.relu(x)
        else:
            # x = torch.tanh(x)
            x = self.gelu(x)
        # try diff norm: batch norm, layernorm
        if only_feat:
            return x
        x = self.dropout(x)
        if not self.regression:
            x = self.norm(x)
        # x = torch.cat((x, fp_features), dim=1)
        x = self.out_proj(x)
        return x


class MultilingualModalUNI(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config, gcn_config, is_regression=False, use_label_weight=False, use_rdkit_feature=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_tasks = config.num_tasks # sider have 27 binary tasks, maybe multi head is useful for multi label classification
        
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
        
        
        if self.use_rdkit_feature:
            assert fp_feature is not None
            logits = self.head(lang_gcn_pooler_output, fp_feature)
        else:
            logits = self.head(lang_gcn_pooler_output)
        
        if labels is None:
            if self.is_regression:
                logits =  self.unnormalize_logits(logits).float()
            if len(logits) == 1:
                logits = logits.reshape(1, -1)
            return logits

        loss = None
        if labels is not None:
            if not self.is_regression:
                if self.num_tasks > 1: # >= 1 for bce loss
                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                    if self.num_tasks == 1:	
                        logits = logits.squeeze()
                    loss = loss_fct(logits, labels)
                    mask = (weight != 0)
                    if self.num_tasks == 1:	
                        mask = mask.squeeze()
                    if self.use_label_weight:
                        bs = labels.shape[0]
                        alpha = weight.sum(axis=0) / bs
                        n_weight = weight / alpha
                        if self.num_tasks == 1:	
                            n_weight = n_weight.squeeze()
                        loss = loss * n_weight
                    else:
                        loss = loss * mask
                    loss = loss.sum() / mask.sum()
                    # loss = loss.mean()
                    # import pdb; pdb.set_trace()
                    
                    
                    # loss_fct = CrossEntropyLoss()
                    # loss = 0
                    # for i, l_ele in enumerate(logits):
                    #     loss += loss_fct(l_ele.view(-1, self.num_labels), labels[:,i:i+1].long().view(-1))
                    #     # import pdb; pdb.set_trace()
                    # loss = loss / self.num_tasks
                    # import pdb; pdb.set_trace()
        
                else:
                    if self.use_label_weight:
                        class_weights = torch.FloatTensor(self.task_weight).to(self.device)
                        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                    else:
                        loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels.long().view(-1)
                    )
            else: # regression
                normalized_labels = self.normalize_logits(labels).float()
                # loss_fct = MSELoss() # todo smooth L1
                loss_fct = SmoothL1Loss()
                loss = loss_fct(logits.view(-1), normalized_labels.view(-1))

        
        
        return [loss]

    def normalize_logits(self, tensor):
        return (tensor - self.norm_mean) / self.norm_std

    def unnormalize_logits(self, tensor):
        return (tensor * self.norm_std) + self.norm_mean


class MultilingualModalUNIDDI(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config, gcn_config, is_regression=False, use_label_weight=False, use_rdkit_feature=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_tasks = config.num_tasks # sider have 27 binary tasks, maybe multi head is useful for multi label classification
        
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
            self.head = RobertaFeatureHead(config, regression=is_regression, dual_input=True)
        else:
            self.head = RobertaHead(config, regression=is_regression, dual_input=True)
        
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

    def forward(
        self,
        left_input_ids=None,
        left_attention_mask=None,
        
        right_input_ids=None,
        right_attention_mask=None,
        
        token_type_ids=None,
        position_ids=None,
        
        left_graph=None,
        right_graph=None,
        # strucpos_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        weight=None,
        output_attentions=None,
        output_hidden_states=None,
        fp_feature = None,
        return_dict=None,
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
        left_graph.to(self.device)
        left_gcn_output = self.gnn(left_graph)
        # concat graph atome embeddings and langua embeddings
        left_gcn_embedding_output = self.gcn_embedding(left_gcn_output[1])
        left_gcn_embedding_output = self.LayerNorm(left_gcn_embedding_output)
        left_gcn_embedding_output = self.dropout(left_gcn_embedding_output)



        gcn_embedding_lst = []	
        batch_size = left_input_ids.shape[0]	
        batch_idx = left_graph.batch
        graph_attention_mask = []
        for bs in range(batch_size):
            gcn_embedding_lst.append(left_gcn_embedding_output[batch_idx == bs])
            atom_num = (batch_idx == bs).sum().item()
            graph_attention_mask.append(torch.tensor([1 for _ in range(atom_num)]).to(self.device))
        
        graph_attention_mask = collate_tokens(graph_attention_mask, pad_idx=0, pad_to_multiple=8)
        graph_attention_mask = graph_attention_mask.to(torch.bool)
        
        left_lang_gcn_outputs, left_lang_gcn_attention_mask = self.lang_roberta(
            left_input_ids,
            attention_mask=left_attention_mask,
            # token_type_ids=lingua['token_type_ids'],
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True if self.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,

            graph_input = gcn_embedding_lst,
            graph_batch = left_graph.batch,
            # graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
            gnn_mask_labels = None,
            graph_attention_mask = graph_attention_mask,
            )
        left_lang_gcn_pooler_output = self.lang_pooler(left_lang_gcn_attention_mask, left_lang_gcn_outputs)
        
        
        right_graph.to(self.device)
        right_gcn_output = self.gnn(right_graph)
        # concat graph atome embeddings and langua embeddings
        right_gcn_embedding_output = self.gcn_embedding(right_gcn_output[1])
        right_gcn_embedding_output = self.LayerNorm(right_gcn_embedding_output)
        right_gcn_embedding_output = self.dropout(right_gcn_embedding_output)


        gcn_embedding_lst = []	
        batch_size = right_input_ids.shape[0]	
        batch_idx = right_graph.batch
        graph_attention_mask = []
        for bs in range(batch_size):
            gcn_embedding_lst.append(right_gcn_embedding_output[batch_idx == bs])
            atom_num = (batch_idx == bs).sum().item()
            graph_attention_mask.append(torch.tensor([1 for _ in range(atom_num)]).to(self.device))
        
        graph_attention_mask = collate_tokens(graph_attention_mask, pad_idx=0, pad_to_multiple=8)
        graph_attention_mask = graph_attention_mask.to(torch.bool)
        
        right_lang_gcn_outputs, right_lang_gcn_attention_mask = self.lang_roberta(
            right_input_ids,
            attention_mask=right_attention_mask,
            # token_type_ids=lingua['token_type_ids'],
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True if self.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,

            graph_input = gcn_embedding_lst,
            graph_batch = right_graph.batch,
            # graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
            gnn_mask_labels = None,
            graph_attention_mask = graph_attention_mask,
            )
        right_lang_gcn_pooler_output = self.lang_pooler(right_lang_gcn_attention_mask, right_lang_gcn_outputs)
        
        lang_gcn_pooler_output = torch.cat((left_lang_gcn_pooler_output, right_lang_gcn_pooler_output), dim=1)
        
        
        if self.use_rdkit_feature:
            assert fp_feature is not None
            logits = self.head(lang_gcn_pooler_output, fp_feature)
        else:
            logits = self.head(lang_gcn_pooler_output)
        
        if labels is None:
            if self.is_regression:
                logits =  self.unnormalize_logits(logits).float()
            if len(logits) == 1:
                logits = logits.reshape(1, -1)
            return logits

        loss = None
        if labels is not None:
            if not self.is_regression:
                if self.num_tasks > 1:
                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                    loss = loss_fct(logits, labels)
                    mask = (weight != 0)
                    if self.use_label_weight:
                        bs = labels.shape[0]
                        alpha = weight.sum(axis=0) / bs
                        n_weight = weight / alpha
                        loss = loss * n_weight
                    else:
                        loss = loss * mask
                    loss = loss.sum() / mask.sum()
                    # loss = loss.mean()
                    # import pdb; pdb.set_trace()
                    
                    
                    # loss_fct = CrossEntropyLoss()
                    # loss = 0
                    # for i, l_ele in enumerate(logits):
                    #     loss += loss_fct(l_ele.view(-1, self.num_labels), labels[:,i:i+1].long().view(-1))
                    #     # import pdb; pdb.set_trace()
                    # loss = loss / self.num_tasks
                    # import pdb; pdb.set_trace()
        
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels.long().view(-1)
                    )
            else: # regression
                normalized_labels = self.normalize_logits(labels).float()
                # loss_fct = MSELoss() # todo smooth L1
                loss_fct = SmoothL1Loss()
                loss = loss_fct(logits.view(-1), normalized_labels.view(-1))

        
        
        return [loss]

    def normalize_logits(self, tensor):
        return (tensor - self.norm_mean) / self.norm_std

    def unnormalize_logits(self, tensor):
        return (tensor * self.norm_std) + self.norm_mean


class MultilingualModalUNIReaction(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config, gcn_config, is_regression=False, use_label_weight=False, use_rdkit_feature=False):
        super().__init__(config)

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
        #self.use_label_weight = use_label_weight

        
        if self.use_rdkit_feature:
            self.head = RobertaFeatureHead(config, regression=is_regression, dual_input=True)
        else:
            self.head = RobertaHead(config, regression=is_regression, dual_input=True)
        
        # self.head = RobertaFeatureHead(config, regression=is_regression)

        
        '''self.is_regression = is_regression
        if is_regression:
            self.register_buffer("norm_mean", torch.tensor(config.norm_mean))
            # Replace any 0 stddev norms with 1
            self.register_buffer(
                "norm_std",
                torch.tensor(
                    [label_std if label_std != 0 else 1 for label_std in config.norm_std]
                ),
            )'''

        self.init_weights()


    def forward(
        self,
        left_input_ids=None, # reactant
        left_attention_mask=None,
        
        right_input_ids=None, # product
        right_attention_mask=None,
        
        token_type_ids=None,
        position_ids=None,
        
        left_graph=None,
        right_graph=None,
        # strucpos_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        weight=None,
        output_attentions=None,
        output_hidden_states=None,
        fp_feature = None,
        return_dict=None,
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        batch_size = left_input_ids.size()[0]
        def calculate_loss(reactant_embeddings, product_embeddings, bs=batch_size):
            margin = 4.0
            dist = torch.cdist(reactant_embeddings, product_embeddings, p=2)
            pos = torch.diag(dist).to(reactant_embeddings.device)
            mask = torch.eye(bs).to(reactant_embeddings.device)
            # if torch.cuda.is_available():
            #     mask = mask.cuda(args.gpu)
            neg = (1 - mask) * dist + mask * margin #args.margin
            neg = torch.relu(margin - neg)
            loss = torch.mean(pos) + torch.sum(neg) / bs / (bs - 1)
            return loss

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # graph_inputs = Batch.from_data_list(graph)
        #left_graph.to(self.device)
        left_gcn_output = self.gnn(left_graph)
        # concat graph atome embeddings and langua embeddings
        left_gcn_embedding_output = self.gcn_embedding(left_gcn_output[1])
        left_gcn_embedding_output = self.LayerNorm(left_gcn_embedding_output)
        left_gcn_embedding_output = self.dropout(left_gcn_embedding_output)
        
        left_lang_gcn_outputs, left_lang_gcn_attention_mask, left_graph_mm_labels = self.lang_roberta(
            left_input_ids,
            attention_mask=left_attention_mask,
            # token_type_ids=lingua['token_type_ids'],
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True if self.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,

            graph_input = left_gcn_embedding_output,
            graph_batch = left_graph.batch,
            graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
            gnn_mask_labels = None,
            )
        left_lang_gcn_pooler_output = self.lang_pooler(left_lang_gcn_attention_mask, left_lang_gcn_outputs)
        
        
        #right_graph.to(self.device)
        right_gcn_output = self.gnn(right_graph)
        # concat graph atome embeddings and langua embeddings
        right_gcn_embedding_output = self.gcn_embedding(right_gcn_output[1])
        right_gcn_embedding_output = self.LayerNorm(right_gcn_embedding_output)
        right_gcn_embedding_output = self.dropout(right_gcn_embedding_output)
        
        right_lang_gcn_outputs, right_lang_gcn_attention_mask, right_graph_mm_labels = self.lang_roberta(
            right_input_ids,
            attention_mask=right_attention_mask,
            # token_type_ids=lingua['token_type_ids'],
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True if self.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,

            graph_input = right_gcn_embedding_output,
            graph_batch = right_graph.batch,
            graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
            gnn_mask_labels = None,
            )
        right_lang_gcn_pooler_output = self.lang_pooler(right_lang_gcn_attention_mask, right_lang_gcn_outputs)
        
        '''lang_gcn_pooler_output = torch.cat((left_lang_gcn_pooler_output, right_lang_gcn_pooler_output), dim=1)
        
        
        if self.use_rdkit_feature:
            assert fp_feature is not None
            logits = self.head(lang_gcn_pooler_output, fp_feature)
        else:
            logits = self.head(lang_gcn_pooler_output)'''
        
        loss = calculate_loss(left_lang_gcn_pooler_output,right_lang_gcn_pooler_output)
        
        return [loss]



class VAE(nn.Module):
    # embedding_dim
    def __init__(self, max_len, vocab_len, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len
        # self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        # self.encoder = nn.Sequential(nn.Linear(max_len * embedding_dim, 2000),
        #                              nn.ReLU(),
        #                              nn.Linear(2000, 1000),
        #                              nn.BatchNorm1d(1000),
        #                              nn.ReLU(),
        #                              nn.Linear(1000, 1000),
        #                              nn.BatchNorm1d(1000),
        #                              nn.ReLU(),
        #                              nn.Linear(1000, latent_dim * 2))
        # self.decoder = nn.Sequential(nn.Linear(latent_dim, 1000),
        #                              nn.BatchNorm1d(1000),
        #                              nn.ReLU(),
        #                              nn.Linear(1000, 1000),
        #                              nn.BatchNorm1d(1000),
        #                              nn.ReLU(),
        #                              nn.Linear(1000, 2000),
        #                              nn.ReLU(),
        #                              nn.Linear(2000, max_len * vocab_len))


        self.decoder = nn.Sequential(nn.Linear(latent_dim, 1000),
                                     nn.LayerNorm(1000, eps=1e-12),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.LayerNorm(1000, eps=1e-12),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, max_len * vocab_len))
    
    def encode(self, x): # x embeddings
        # x = self.encoder(self.embedding(x).view((len(x), -1))).view((-1, 2, self.latent_dim))
        x = x.view((-1, 2, self.latent_dim))
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, x):
        return F.log_softmax(self.decoder(x).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), z, mu, log_var
    
    def loss_function(self, pred, target, mu, log_var, batch_size, p):
        nll = F.nll_loss(pred, target)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (batch_size * pred.shape[1])
        return (1 - p) * nll + p * kld, nll, kld
    
    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var = self(train_batch)
        p = 0.1
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), train_batch.flatten(), mu, log_var, len(train_batch), p)
        self.log('train_loss', loss)
        self.log('train_nll', nll)
        self.log('train_kld', kld)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var = self(val_batch)
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), val_batch.flatten(), mu, log_var, len(val_batch), 0.5)
        self.log('val_loss', loss)
        self.log('val_nll', nll)
        self.log('val_kld', kld)
        self.log('val_mu', torch.mean(mu))
        self.log('val_logvar', torch.mean(log_var))
        return loss



class T5VAE(T5PreTrainedModel):
    def __init__(self, config, latent_dim, pooling_strategy='max'):
        super().__init__(config)
        self.latent_dim = latent_dim
        self.mu = nn.Linear(config.d_model, latent_dim, bias=False)
        self.logvar = nn.Linear(config.d_model, latent_dim, bias=False)
        self.embed_size_per_head = config.d_kv
        self.model_dim = config.d_model
        self.memory_projection = nn.Linear(
            latent_dim,
            config.num_decoder_layers * config.num_heads * self.embed_size_per_head,
            bias=False,
        )
        self.pooling_strategy = pooling_strategy
        config.decoder_start_token_id = config.vocab_size - 1 # last(108) as padding idx, 0 as the ending
        self.padding_idx = config.decoder_start_token_id
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = T5Stack(decoder_config, self.shared)

        predict_size = config.vocab_size - 1 # do not predict the vocab size
        self.lm_head = nn.Linear(config.d_model, predict_size, bias=False)

        # Initialize weights and apply final processing
        self.init_weights()

    
    def forward(
            self,
            input_ids=None, # selfies input ids
            attention_mask=None,
            # decoder_input_ids=None, # selfies input ids, shift away
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None, # if training, cannot be none
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sampled_z=None,
            only_z=False,
        ):

        z, mu, logvar = None, None, None
        if sampled_z is not None: # inference
            z = sampled_z
            encoder_outputs = BaseModelOutput(
                last_hidden_state=None,
                hidden_states=None,
                attentions=None,
            )
        if encoder_outputs is not None: # training
            pooled = self.pool(encoder_outputs.hidden_states)
            z, mu, logvar = self.calculate_latent(pooled)
            if only_z:
                return z
        # elif encoder_outputs is None:
        #     # Convert encoder inputs in embeddings if needed
        #     encoder_outputs = self.run_encoder(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         inputs_embeds=inputs_embeds,
        #         head_mask=head_mask,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        #     pooled = self.pool(encoder_outputs.hidden_states)
        #     z, mu, logvar = self.calculate_latent(pooled)
        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        # hidden_states = encoder_outputs[0]

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        # if (
        #     labels is not None
        #     and decoder_input_ids is None
        #     and decoder_inputs_embeds is None
        # ):
            # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(input_ids)

        if past_key_values is None:
            past_key_values = self.build_past(z)

       

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        sequence_output = decoder_outputs[0]


        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     # loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        #     pass

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # out = Seq2SeqLMOutput(
        #     # loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )
        # out.mu = mu
        # out.logvar = logvar
        # out.z = z

        # calculate loss
        loss_dict = {}
        recon_loss = self.reconstruction_loss(lm_logits, input_ids)
        reg_loss = self.regularization_loss(mu, logvar, True)
        loss = recon_loss + 0.001 * reg_loss.mean()
        loss_dict['recon_loss'] = recon_loss.item()
        loss_dict['reg_loss'] = reg_loss.mean().item()
        loss_dict['loss'] = loss.item()
        print(loss_dict)
        return loss

    def reconstruction_loss(self, x, target):
        target[target==self.padding_idx] = -100
        loss = F.cross_entropy(
            x.transpose(1, 2),
            target,
            ignore_index=-100,
        )
        return loss

    def regularization_loss(self, mu, logvar, training=False):
        dimensionwise_loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
        min_z = 0.5
        if min_z:
            dimensionwise_loss[dimensionwise_loss < min_z] = min_z
        loss = dimensionwise_loss.sum(-1)
        return loss


    def pool(self, x):
        # Shape of x - (layer_count, batch_size, seq_length, hidden_size)
        x = torch.stack(x[1:])
        x = x.transpose(0, 1)
        if self.pooling_strategy == "mean":
            return x[:, -1, :, :].mean(dim=1)
        elif self.pooling_strategy == "max":
            return torch.max(x[:, -1, :, :], dim=1)[0]  # Pool from last layer.
        else:
            raise Exception("Wrong pooling strategy!")

    def calculate_latent(self, pooled):
        mu, logvar = self.mu(pooled), self.logvar(pooled)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def build_past(self, z):
        projection = self.memory_projection(z)
        cross_attn = projection.reshape(
            self.config.num_decoder_layers,
            projection.shape[0],
            self.config.num_heads,
            1,
            self.embed_size_per_head,
        )
        past_key_values = tuple((ca, ca) for ca in cross_attn)
        return past_key_values

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        """
        assert (
            logits.dim() == 1
        )  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits


    def generate(self, starter_tokens, eos_token_id, **kwargs):
        generated = torch.tensor([starter_tokens]).unsqueeze(0).to(self.device)
        output, encoder_outputs = None, None

        only_prob = kwargs.get("only_prob") if "only_prob" in kwargs else False

        # debug
        # only_prob = True


        if only_prob:
            prob_lst = []
        
        max_seq_len = kwargs.get("max_seq_len") if "max_seq_len" in kwargs else 100

        while generated.shape[1] <= max_seq_len:
            sampled_z = kwargs.get("sampled_z") if output is None else None

            # with torch.no_grad():
            past_key_values = self.build_past(sampled_z)
            # Decode
            decoder_outputs = self.decoder(
                input_ids=generated,
                attention_mask=None,
                inputs_embeds=None,
                past_key_values=past_key_values,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                head_mask=None,
                cross_attn_head_mask=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=True,
            )

            sequence_output = decoder_outputs[0]

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim ** -0.5)

            lm_logits = self.lm_head(sequence_output)

                    # logits.append(lm_logits[])

                # output = model.t5(
                #     input_ids=kwargs.get("input_ids"),
                #     attention_mask=kwargs.get("attention_mask"),
                #     # attention_mask=torch.ones((generated.shape[0], generated.shape[1] + 1)),
                #     # encoder_hidden_states=None, #new_encoder_hidden_states,  # Modified.
                #     # encoder_attention_mask=None, #new_attention_mask,  # Modified.
                #     # attention_mask=encoder_mask,
                #     encoder_outputs=encoder_outputs,
                #     decoder_input_ids=generated[:, -1].unsqueeze(0),
                #     # encoder_hidden_states=encoder_outputs[0],  # Modified.
                #     # encoder_attention_mask=attention_mask,  # Modified.
                #     # head_mask=kwargs.get("decoder_head_mask"),
                #     # cross_attn_head_mask=kwargs.get("cross_attn_head_mask"),
                #     past_key_values=output.past_key_values if output else None,
                #     # inputs_embeds=decoder_inputs_embeds,
                #     use_cache=True,
                #     # output_attentions=output_attentions,
                #     output_hidden_states=True,
                #     return_dict=True,
                #     sampled_z=sampled_z,
                # )

            temperature = kwargs.get("temperature") if "temperature" in kwargs else 1.0
            top_k = kwargs.get("top_k") if "top_k" in kwargs else 0
            top_p = kwargs.get("top_p") if "top_p" in kwargs else 0

            logits = lm_logits[0, -1, :] / temperature
            filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probabilities = F.softmax(filtered_logits, dim=-1)
            
            next_token_id = torch.multinomial(probabilities, 1)
            # next_token_id = torch.argmax(filtered_logits, dim=-1).unsqueeze(0)

            generated = torch.cat((generated, next_token_id.unsqueeze(0)), dim=1)
            if only_prob:
                prob_lst.append(probabilities)
                continue
            if next_token_id.item() == eos_token_id and not only_prob:
                break
            
        if only_prob:
            return torch.stack(prob_lst, dim=0)


        return generated[:,1:]




    pass

class MultilingualModalUNIVAE(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config, gcn_config):
        super().__init__(config)
        
        use_t5 = True
        self.latent_dim = config.latent_dim
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
        
        self.use_t5 = use_t5
        self.self_vocab_len = config.self_vocab_len
        if use_t5:
            t5_config = T5Config()
            t5_config.d_model = config.hidden_size
            t5_config.is_decoder = True
            t5_config.is_encoder_decoder = False
            t5_config.num_layers = config.num_hidden_layers
            t5_config.vocab_size = config.self_vocab_len
            self.vae = T5VAE(t5_config, latent_dim=self.latent_dim)
        else:
            self.vae = VAE(config.predict_max_len, config.self_vocab_len, config.hidden_size)

        
        # self.use_rdkit_feature = use_rdkit_feature
        # self.use_label_weight = use_label_weight

        
        # if self.use_rdkit_feature:
        #     self.head = RobertaFeatureHead(config, regression=is_regression)
        # else:
        #     self.head = RobertaHead(config, regression=is_regression)
        
        # # self.head = RobertaFeatureHead(config, regression=is_regression)

        
        # self.is_regression = is_regression
        # if is_regression:
        #     self.register_buffer("norm_mean", torch.tensor(config.norm_mean))
        #     # Replace any 0 stddev norms with 1
        #     self.register_buffer(
        #         "norm_std",
        #         torch.tensor(
        #             [label_std if label_std != 0 else 1 for label_std in config.norm_std]
        #         ),
        #     )

        self.emb_dense = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.layer_norm = nn.LayerNorm(config.hidden_size * 2, eps=config.layer_norm_eps)


        self.init_weights()
        
    #     self.task_weight = None
    
    # def set_task_weight(self, task_weight):
    #     self.task_weight = task_weight

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
        target_encoding=None,
        fp_feature = None,
        return_dict=None,
        num_mols=0,
        only_z=False,
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
        
        

        if target_encoding == None and not only_z:
            # eval mode
            if self.use_t5:
                z = torch.randn((1, self.latent_dim), device=self.device)
                logits = self.vae.generate(starter_tokens=108, eos_token_id=0, sampled_z=z)
                return logits[0]
            
            else:
                lang_gcn_pooler_output = torch.randn((num_mols, self.config.hidden_size * 2), device=self.device)
                out, z, mu, log_var = self.vae(lang_gcn_pooler_output)
                with torch.no_grad():
                    x = torch.exp(out)
                    # smiles = [one_hot_to_smiles(hot) for hot in x]
                    # return smiles
                    return x
        else:
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
                output_hidden_states=True,
                return_dict=True,

                graph_input = gcn_embedding_lst,
                graph_batch = graph.batch,
                # graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
                gnn_mask_labels = None,
                graph_attention_mask = graph_attention_mask,
                )

            
            if self.use_t5:
                # input_ids=None, # selfies input ids
                # attention_mask=None,
                # # decoder_input_ids=None, # selfies input ids, shift away
                # decoder_attention_mask=None,
                # head_mask=None,
                # decoder_head_mask=None,
                # cross_attn_head_mask=None,
                # encoder_outputs=None, # if training, cannot be none
                # past_key_values=None,
                # inputs_embeds=None,
                # decoder_inputs_embeds=None,
                # labels=None,
                # use_cache=None,
                # output_attentions=None,
                # output_hidden_states=None,
                # return_dict=None,
                # sampled_z=None,
                if only_z:
                    z = self.vae(input_ids=target_encoding, encoder_outputs=lang_gcn_outputs, only_z=only_z)
                    return z

                loss = self.vae(input_ids=target_encoding, encoder_outputs=lang_gcn_outputs)
                return [loss]
            else:
                lang_gcn_pooler_output = self.lang_pooler(lang_gcn_attention_mask, lang_gcn_outputs)

                hidden_embedding = self.emb_dense(lang_gcn_pooler_output)
                out, z, mu, log_var = self.vae(hidden_embedding)
                loss, nll, kld = self.vae.loss_function(out.reshape((-1, self.self_vocab_len)), target_encoding.flatten(), mu, log_var, len(target_encoding), 0.5)
                return [loss]

    def normalize_logits(self, tensor):
        return (tensor - self.norm_mean) / self.norm_std

    def unnormalize_logits(self, tensor):
        return (tensor * self.norm_std) + self.norm_mean
