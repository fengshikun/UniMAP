import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from gcn import DeeperGCN, AtomHead
import numpy as np

# from gformer.graphormer import GraphormerEncoder


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, vocab_size_sp=None, graph_hidden_size=None, atom_vocab_size=None):
        super().__init__()
        first_hidden_size = config.hidden_size
        if graph_hidden_size is not None:
            first_hidden_size += graph_hidden_size
        self.dense = nn.Linear(first_hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if vocab_size_sp is not None:
            vocab_size = vocab_size_sp
        elif atom_vocab_size is not None:
            vocab_size = config.vocab_size + atom_vocab_size
        else:
            vocab_size = config.vocab_size

        self.decoder = nn.Linear(config.hidden_size, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class MLPLayerMoco(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn = nn.BatchNorm1d(config.hidden_size)
        # self.bn = nn.LayerNorm(config.hidden_size)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, features, **kwargs):
        x = self.dense1(features)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dense2(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp



class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_init_2(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = config.pooler_type
    cls.pooler = Pooler(config.pooler_type)
    cls.smiles_pooler = Pooler(config.pooler_type)
    if config.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
        cls.smiles_mlp = MLPLayer(config)
    
    cls.sim = Similarity(config.temp)
    cls.init_weights()


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

def convert_input(inputs, device):
    if inputs is None:
        return inputs
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def cl_forward_2(cls,
    encoder_iupac,
    encoder_smiles,
    iupac_input,
    smiles_input,
    iupac_hard_input=None,
    # input_ids_=None,
    # attention_mask=None,
    # token_type_ids=None,
    # position_ids=None,
    # head_mask=None,
    # inputs_embeds=None,
    # labels=None,
    # output_attentions=None,
    # output_hidden_states=None,
    # return_dict=None,
    # mlm_input_ids=None,
    # mlm_labels=None,
    ):
    
    return_dict = {}
    
    iupac_outputs = encoder_iupac(
        iupac_input['input_ids'],
        attention_mask=iupac_input['attention_mask'],
        token_type_ids=iupac_input['token_type_ids'],
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=True if cls.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    
    smiles_outputs = encoder_smiles(
        smiles_input['input_ids'],
        attention_mask=smiles_input['attention_mask'],
        token_type_ids=smiles_input['token_type_ids'],
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=True if cls.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    
    if iupac_hard_input:
        iupac_hard_outputs = encoder_iupac(
            iupac_hard_input['input_ids'],
            attention_mask=iupac_hard_input['attention_mask'],
            token_type_ids=iupac_hard_input['token_type_ids'],
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True if cls.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
     
    # MLM auxiliary objective
    if 'mlm_input_ids' in iupac_input:
        iupac_mlm_outputs = encoder_iupac(
            iupac_input['mlm_input_ids'],
            attention_mask=iupac_input['attention_mask'],
            token_type_ids=iupac_input['token_type_ids'],
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True if cls.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        mlm_labels = iupac_input['mlm_labels']
        prediction_scores = cls.iupac_lm_head(iupac_mlm_outputs.last_hidden_state)
        mlm_loss_iupac = cls.loss_mlm(prediction_scores.view(-1, cls.iupac_config.vocab_size), mlm_labels.view(-1))
        return_dict['mlm_loss_iupac'] = mlm_loss_iupac
        
    if 'mlm_input_ids' in smiles_input:
        if smiles_input['mlm_input_ids'].max().item() >= cls.smiles_config.vocab_size:
            print("error") 
        smiles_mlm_outputs = encoder_smiles(
            smiles_input['mlm_input_ids'],
            attention_mask=smiles_input['attention_mask'],
            token_type_ids=smiles_input['token_type_ids'],
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True if cls.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        mlm_labels = smiles_input['mlm_labels']
        prediction_scores = cls.smiles_lm_head(smiles_mlm_outputs.last_hidden_state)
        mlm_loss_smiles = cls.loss_mlm(prediction_scores.view(-1, cls.smiles_config.vocab_size), mlm_labels.view(-1))
        return_dict['mlm_loss_smiles'] = mlm_loss_smiles
    # contrastive learning part
    
   
    iupac_pooler_output = cls.pooler(iupac_input['attention_mask'], iupac_outputs)
    iupac_hard_pooler_output = cls.pooler(iupac_hard_input['attention_mask'], iupac_hard_outputs)
    smiles_pooler_output = cls.pooler(smiles_input['attention_mask'], smiles_outputs)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    iupac_number = iupac_pooler_output.shape[0]
    if cls.pooler_type == "cls":
        # concat iupac and iupac_hard
        iupac_fea_input = torch.cat([iupac_pooler_output, iupac_hard_pooler_output], dim=0)
        iupac_fea_output = cls.mlp(iupac_fea_input)
        smiles_fea_output = cls.smiles_mlp(smiles_pooler_output)
    
    iupuc_all_number = iupac_fea_output.shape[0]
    
    # concat smiles and iupac_hard
    smiles_iupac_hard_output = torch.cat([smiles_fea_output, iupac_fea_output[iupac_number:]], dim=0)
    
    
    cos_sim = cls.sim(iupac_fea_output.unsqueeze(1), smiles_iupac_hard_output.unsqueeze(0))
    
    # pick rows according to the iupac_fea
    iupac_rows = cos_sim[:iupac_number, :]
    # pick rows according to the smiles
    smiles_rows = cos_sim[:, :iupac_number].t()
    labels = torch.arange(iupac_number).long().to(cls.device)
    
    ctr_loss_iupac = cls.loss_mlm(iupac_rows, labels)
    ctr_loss_smiles = cls.loss_mlm(smiles_rows, labels)
    
    return_dict['ctr_loss_iupac'] = ctr_loss_iupac
    return_dict['ctr_loss_smiles'] = ctr_loss_smiles
    
    loss = 0
    for _, item_loss in return_dict.items():
        loss += item_loss
    
    return_dict['loss'] = loss
    
    # print('finish')
    return return_dict
    # pass



def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )






class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_save = []

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )




class DualRobertaForCL(RobertaPreTrainedModel):
    
    def __init__(self, iupac_config, smiles_config):
        super(DualRobertaForCL, self).__init__(iupac_config)
        self.iupac_roberta = RobertaModel(iupac_config, add_pooling_layer=False)
        self.smiles_roberta = RobertaModel(smiles_config, add_pooling_layer=False)
        
        self.iupac_lm_head = RobertaLMHead(iupac_config)
        
        self.smiles_lm_head = RobertaLMHead(smiles_config)
        
        self.iupac_config = iupac_config
        self.smiles_config = smiles_config
        self.loss_mlm = nn.CrossEntropyLoss()
        
        cl_init_2(self, iupac_config)
        
    
    # def _init_weights(self, module):
    #     """Initialize the weights"""
    #     if isinstance(module, nn.Linear):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    
    def forward(self, smiles=None, iupac=None, iupac_hard=None):
        # print("forward")
        iupac_input = convert_input(iupac, self.device)
        iupac_hard_input = convert_input(iupac_hard, self.device)
        smiles_input = convert_input(smiles, self.device)
        return cl_forward_2(self, self.iupac_roberta, self.smiles_roberta, iupac_input, smiles_input, iupac_hard_input=iupac_hard_input)
        pass


class IupacRobertaForCL(RobertaPreTrainedModel):
    
    def __init__(self, iupac_config):
        super(IupacRobertaForCL, self).__init__(iupac_config)
        self.iupac_roberta = RobertaModel(iupac_config, add_pooling_layer=False)
        # self.smiles_roberta = RobertaModel(smiles_config, add_pooling_layer=False)
        
        self.iupac_lm_head = RobertaLMHead(iupac_config)
        
        # self.smiles_lm_head = RobertaLMHead(smiles_config)
        
        self.iupac_config = iupac_config
        # self.smiles_config = smiles_config
        self.loss_mlm = nn.CrossEntropyLoss()
        
        self.pooler_type = iupac_config.pooler_type
        self.pooler = Pooler(iupac_config.pooler_type)
        if iupac_config.pooler_type == "cls":
            self.mlp = MLPLayer(self.config)
        
        self.sim = Similarity(self.config.temp)
        self.init_weights()
        
    
    # def _init_weights(self, module):
    #     """Initialize the weights"""
    #     if isinstance(module, nn.Linear):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    
    def forward(self, iupac=None, iupac_hard=None):
        # print("forward")
        # concat iupac and iupac_pos to iupac
        iupac_input = convert_input(iupac, self.device)
        iupac_hard_input = convert_input(iupac_hard, self.device)
        # iupac_pos_input = convert_input(iupac_pos, self.device)
        return_dict = {}
        
        if iupac_hard_input:
            iupac_outputs = self.iupac_roberta(
                iupac_input['input_ids'],
                attention_mask=iupac_input['attention_mask'],
                token_type_ids=iupac_input['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )

            # ctr loss
            iupac_hard_outputs = self.iupac_roberta(
                iupac_hard_input['input_ids'],
                attention_mask=iupac_hard_input['attention_mask'],
                token_type_ids=iupac_hard_input['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )
            iupac_pooler_output = self.pooler(iupac_input['attention_mask'], iupac_outputs)
            iupac_hard_pooler_output = self.pooler(iupac_hard_input['attention_mask'], iupac_hard_outputs)

            # If using "cls", we add an extra MLP layer
            # (same as BERT's original implementation) over the representation.
            iupac_pair_number = iupac_pooler_output.shape[0]
            iupac_fea_input = torch.cat([iupac_pooler_output, iupac_hard_pooler_output], dim=0)
            if self.pooler_type == "cls":
                # concat iupac and iupac_hard
                iupac_fea_input = self.mlp(iupac_fea_input)
            

            iupac_org_emb = iupac_fea_input[:iupac_pair_number]

            iupac_emb = iupac_org_emb[::2]
            iupac_pos_emb = iupac_org_emb[1::2]
            iupac_hard_emb = iupac_fea_input[iupac_pair_number:, :]

            z1 = iupac_emb
            z2 = iupac_pos_emb
            z3 = iupac_hard_emb
            
            if dist.is_initialized() and self.training:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

                # Dummy vectors for allgather
                z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
                z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
                # Allgather
                dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
                dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

                # Since allgather results do not have gradients, we replace the
                # current process's corresponding embeddings with original tensors
                z1_list[dist.get_rank()] = z1
                z2_list[dist.get_rank()] = z2
                # Get full batch embeddings: (bs x N, hidden)
                z1 = torch.cat(z1_list, 0)
                z2 = torch.cat(z2_list, 0)


            cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            # Hard negative
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

            labels = torch.arange(cos_sim.size(0)).long().to(self.device)
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(cos_sim, labels)            
            return_dict['ctr_loss'] = loss
        

        else:
            # mlm loss
            assert 'mlm_input_ids' in iupac_input
            iupac_mlm_outputs = self.iupac_roberta(
                iupac_input['mlm_input_ids'],
                attention_mask=iupac_input['attention_mask'],
                token_type_ids=iupac_input['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )
            mlm_labels = iupac_input['mlm_labels']
            prediction_scores = self.iupac_lm_head(iupac_mlm_outputs.last_hidden_state)
            mlm_loss_iupac = self.loss_mlm(prediction_scores.view(-1, self.iupac_config.vocab_size), mlm_labels.view(-1))
            return_dict['mlm_loss_iupac'] = mlm_loss_iupac
        
        
        
        loss = 0
        for _, item_loss in return_dict.items():
            loss += item_loss
        
        if torch.isnan(loss).sum():
            torch.save({'z1':z1, 'z2':z2,'z3':z3,'iupac_org_emb': iupac_org_emb, 'iupac_hard_emb': iupac_hard_emb,
            'iupac_input': iupac_input['input_ids'], 'iupac_hard_input': iupac_hard_input['input_ids']}, 'emb_info_{}.pt'.format(dist.get_rank()))
            torch.save(cos_sim, 'cos_sim_rank_{}.pt'.format(dist.get_rank()))
            import pdb;pdb.set_trace()
            print("debug") 


        return_dict['loss'] = loss
        
        
        # print('finish')
        return return_dict
        # pass




class MultilingualModel(RobertaPreTrainedModel):
    def __init__(self, multilingua_config, gcn_config, atom_vocab_size, K=65536, T=0.07):
        super(MultilingualModel, self).__init__(multilingua_config)
        
        self.multilingua_config = multilingua_config
        self.lang_roberta = RobertaModel(multilingua_config, add_pooling_layer=True)
        self.gnn = DeeperGCN(gcn_config)
        
        self.lm_head = RobertaLMHead(multilingua_config, graph_hidden_size=gcn_config['gnn_embed_dim'] * 2)
        self.atom_head = AtomHead(multilingua_config.hidden_size + gcn_config['gnn_embed_dim'], 
                                  atom_vocab_size, 'gelu', norm=gcn_config['gnn_norm'])
        self.lang_pooler = Pooler(multilingua_config.pooler_type)
        
        self.config = multilingua_config
        self.gcn_config = gcn_config
        self.loss_mlm = nn.CrossEntropyLoss()
        
        
        # corss modal encoder
        # self.cross_encoder = nn.ModuleList([RobertaLayer(multilingua_config) for _ in range(multilingua_config.num_hidden_layers // 4)])
        # cl_init_2(self, iupac_config)
        hidden_size = multilingua_config.hidden_size
        
        # head for the lang
        self.lang_mlp = MLPLayerMoco(multilingua_config)
        # head for the gcn
        self.gcn_mlp = MLPLayerMoco(multilingua_config)
        # gcn embedding converter
        self.gcn_embedding = nn.Linear(gcn_config['gnn_embed_dim'], multilingua_config.hidden_size, bias=True)
        self.LayerNorm = torch.nn.LayerNorm(multilingua_config.hidden_size, eps=1e-12)
        
        
        # memory bank for the lang
        self.register_buffer("lang_queue", torch.randn(hidden_size, K))
        self.lang_queue = nn.functional.normalize(self.lang_queue, dim=0)
        self.register_buffer("lang_queue_ptr", torch.zeros(1, dtype=torch.long))

        # memory bank for the gnn
        self.register_buffer("gcn_queue", torch.randn(hidden_size, K))
        self.gcn_queue = nn.functional.normalize(self.gcn_queue, dim=0)
        self.register_buffer("gcn_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.T = T
        self.K = K
        
    
    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output            
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        # gather keys before updating queue
        keys = self.concat_all_gather(keys)
        # print('keys shape: {}'.format(keys.shape[0]))
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        queue_ptr[0] = ptr
    
    
    
    def forward(self, lingua=None, graph=None, gm_masked_labels=None):
        return_dict = {}
        if lingua['input_ids'].device != self.device:
            lingua['input_ids'] = lingua['input_ids'].to(self.device)
            if 'mlm_input_ids' in lingua:
                lingua['mlm_input_ids'] = lingua['mlm_input_ids'].to(self.device)
                lingua['mlm_labels'] = lingua['mlm_labels'].to(self.device)
            lingua['attention_mask'] = lingua['attention_mask'].to(self.device)
        graph.to(self.device)
        
        if gm_masked_labels is not None:
            # print("mask graph")
            lingua_outputs = self.lang_roberta(
                lingua['input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            lang_pooler_output = self.lang_pooler(lingua['attention_mask'], lingua_outputs)
            gcn_output = self.gnn(graph)
            # split the lang_pooler_output into smiles and iupacs
            smiles_cls = lang_pooler_output[::2]
            iupac_cls = lang_pooler_output[1::2]
            graph_batch = graph.batch
            graph_smiles_logits = self.atom_head(node_features=gcn_output[1], cls_features=smiles_cls[graph_batch])
            graph_iupac_logits = self.atom_head(node_features=gcn_output[1], cls_features=iupac_cls[graph_batch])
            graph_smiles_mask_loss = self.loss_mlm(graph_smiles_logits, gm_masked_labels)
            graph_iupac_mask_loss = self.loss_mlm(graph_iupac_logits, gm_masked_labels)
            return_dict['graph_smiles_mask_loss'] = graph_smiles_mask_loss
            return_dict['graph_iupac_mask_loss'] = graph_iupac_mask_loss
        
        elif 'mlm_input_ids' in lingua: # lang mask resume with graph assist
            # print("mask lang")
            gcn_output = self.gnn(graph)
            lingua_outputs = self.lang_roberta(
                lingua['mlm_input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            lang_pooler_output = self.lang_pooler(lingua['attention_mask'], lingua_outputs)
            graph_emb = gcn_output[0]
            token_num = lingua['mlm_input_ids'].shape[1]
            graph_emb = graph_emb.repeat(1, 1, token_num).reshape(graph_emb.shape[0], -1, graph_emb.shape[1])
            mlm_labels = lingua['mlm_labels']
            graph_emb_concat = torch.zeros_like(lingua_outputs.last_hidden_state)
            graph_emb_concat[::2] = graph_emb
            graph_emb_concat[1::2] = graph_emb
            features = torch.cat((lingua_outputs.last_hidden_state, graph_emb_concat), dim=2)
            prediction_scores = self.lm_head(features)
            mlm_loss_lang = self.loss_mlm(prediction_scores.view(-1, self.multilingua_config.vocab_size), mlm_labels.view(-1))
            return_dict['mlm_loss_lang'] = mlm_loss_lang
        
        else:
            # print("contrastive learning")
            lingua_outputs = self.lang_roberta(
                lingua['input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            lang_pooler_output = self.lang_pooler(lingua['attention_mask'], lingua_outputs)
            gcn_output = self.gnn(graph)
            graph_emb = gcn_output[0]
            # with torch.no_grad():
            
            lang_project = self.lang_mlp(lang_pooler_output)
            graph_project = self.gcn_mlp(graph_emb)
            
            lang_emb_norm = nn.functional.normalize(lang_project, dim=1)  
            graph_emb_norm = nn.functional.normalize(graph_project, dim=1)
            # smiles_emb_norm = lang_emb_norm[::2]
            # iupac_emb_norm = lang_emb_norm[1::2]
            graph_emb_concat = torch.zeros_like(lang_emb_norm)
            graph_emb_concat[::2] = graph_emb_norm
            graph_emb_concat[1::2] = graph_emb_norm
            
            l_pos = torch.einsum('nc,nc->n', [lang_emb_norm, graph_emb_concat]).unsqueeze(-1)
            l_neg_lang_graph = torch.einsum('nc,ck->nk', [lang_emb_norm, self.gcn_queue.clone().detach()])
            l_neg_graph_lang = torch.einsum('nc,ck->nk', [graph_emb_concat, self.lang_queue.clone().detach()])
            
            lang_logits = torch.cat([l_pos, l_neg_lang_graph], dim=1)
            graph_logits = torch.cat([l_pos, l_neg_graph_lang], dim=1)
            
            lang_logits /= self.T
            graph_logits /= self.T
            labels = torch.zeros(lang_logits.shape[0], dtype=torch.long).cuda()
            
            
            
            
            self._dequeue_and_enqueue(graph_emb_concat, self.gcn_queue, self.gcn_queue_ptr)
            self._dequeue_and_enqueue(lang_emb_norm, self.lang_queue, self.lang_queue_ptr)
            
            loss_lang_ctr = self.loss_mlm(lang_logits, labels)
            loss_graph_ctr = self.loss_mlm(graph_logits, labels)
            return_dict['loss_lang_ctr'] = loss_lang_ctr
            return_dict['loss_graph_ctr'] = loss_graph_ctr
            
        loss = 0
        for _, item_loss in return_dict.items():
            loss += item_loss
        return_dict['loss'] = loss
        return return_dict



class MultilingualModelSplit(RobertaPreTrainedModel):
    def __init__(self, iupac_config, smiles_config, gcn_config, atom_vocab_size, K=65536, T=0.07):
        super(MultilingualModelSplit, self).__init__(iupac_config)
        
        self.iupac_config = iupac_config
        self.smiles_config = smiles_config
        self.iupac_roberta = RobertaModel(iupac_config, add_pooling_layer=True)
        self.iupac_head = RobertaLMHead(iupac_config, graph_hidden_size=gcn_config['gnn_embed_dim'] * 2)

        self.smiles_roberta = RobertaModel(smiles_config, add_pooling_layer=True)
        self.smiles_head = RobertaLMHead(smiles_config, graph_hidden_size=gcn_config['gnn_embed_dim'] * 2)

        self.gnn = DeeperGCN(gcn_config)
        
        
        self.atom_head = AtomHead(self.iupac_config.hidden_size + gcn_config['gnn_embed_dim'], 
                                  atom_vocab_size, 'gelu', norm=gcn_config['gnn_norm'])
        self.lang_pooler = Pooler(self.iupac_config.pooler_type)
        
        # self.config = multilingua_config
        self.gcn_config = gcn_config
        self.loss_mlm = nn.CrossEntropyLoss()
        
        
        # corss modal encoder
        # self.cross_encoder = nn.ModuleList([RobertaLayer(multilingua_config) for _ in range(multilingua_config.num_hidden_layers // 4)])
        # cl_init_2(self, iupac_config)
        
        
        # head for the lang
        self.iupac_mlp = MLPLayerMoco(self.iupac_config)
        self.smiles_mlp = MLPLayerMoco(self.smiles_config)
        # head for the gcn
        self.gcn_mlp = MLPLayerMoco(self.iupac_config)
        
        
        hidden_size = self.iupac_config.hidden_size
        # memory bank for the lang
        self.register_buffer("iupac_queue", torch.randn(hidden_size, K))
        self.iupac_queue = nn.functional.normalize(self.iupac_queue, dim=0)
        self.register_buffer("iupac_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("smiles_queue", torch.randn(hidden_size, K))
        self.smiles_queue = nn.functional.normalize(self.smiles_queue, dim=0)
        self.register_buffer("smiles_queue_ptr", torch.zeros(1, dtype=torch.long))

        # memory bank for the gnn
        self.register_buffer("gcn_queue", torch.randn(hidden_size, K))
        self.gcn_queue = nn.functional.normalize(self.gcn_queue, dim=0)
        self.register_buffer("gcn_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.T = T
        self.K = K
        
    
    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output            
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        # gather keys before updating queue
        keys = self.concat_all_gather(keys)
        # print('keys shape: {}'.format(keys.shape[0]))
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        queue_ptr[0] = ptr
    
    
    
    def forward(self, iupac=None, smiles=None, graph=None, gm_masked_labels=None):
        return_dict = {}
        if iupac['input_ids'].device != self.device:
            iupac['input_ids'] = iupac['input_ids'].to(self.device)
            if 'mlm_input_ids' in iupac:
                iupac['mlm_input_ids'] = iupac['mlm_input_ids'].to(self.device)
                iupac['mlm_labels'] = iupac['mlm_labels'].to(self.device)
            iupac['attention_mask'] = iupac['attention_mask'].to(self.device)
        
        if smiles['input_ids'].device != self.device:
            smiles['input_ids'] = smiles['input_ids'].to(self.device)
            if 'mlm_input_ids' in smiles:
                smiles['mlm_input_ids'] = smiles['mlm_input_ids'].to(self.device)
                smiles['mlm_labels'] = smiles['mlm_labels'].to(self.device)
            smiles['attention_mask'] = smiles['attention_mask'].to(self.device)

        graph.to(self.device)
        
        if gm_masked_labels is not None:
            # print("mask graph")
            iupac_outputs = self.iupac_roberta(
                iupac['input_ids'],
                attention_mask=iupac['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            iupac_pooler_output = self.lang_pooler(iupac['attention_mask'], iupac_outputs)

            smiles_outputs = self.smiles_roberta(
                smiles['input_ids'],
                attention_mask=smiles['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.smiles_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            smiles_pooler_output = self.lang_pooler(smiles['attention_mask'], smiles_outputs)


            gcn_output = self.gnn(graph)
            # split the lang_pooler_output into smiles and iupacs
            smiles_cls = smiles_pooler_output
            iupac_cls = iupac_pooler_output
            graph_batch = graph.batch
            graph_smiles_logits = self.atom_head(node_features=gcn_output[1], cls_features=smiles_cls[graph_batch])
            graph_iupac_logits = self.atom_head(node_features=gcn_output[1], cls_features=iupac_cls[graph_batch])
            graph_smiles_mask_loss = self.loss_mlm(graph_smiles_logits, gm_masked_labels)
            graph_iupac_mask_loss = self.loss_mlm(graph_iupac_logits, gm_masked_labels)
            return_dict['graph_smiles_mask_loss'] = graph_smiles_mask_loss
            return_dict['graph_iupac_mask_loss'] = graph_iupac_mask_loss
        
        elif 'mlm_input_ids' in iupac: # lang mask resume with graph assist
            # print("mask lang")
            gcn_output = self.gnn(graph)
            iupac_outputs = self.iupac_roberta(
                iupac['mlm_input_ids'],
                attention_mask=iupac['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            iupac_pooler_output = self.lang_pooler(iupac['attention_mask'], iupac_outputs)

            smiles_outputs = self.smiles_roberta(
                smiles['mlm_input_ids'],
                attention_mask=smiles['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.smiles_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            smiles_pooler_output = self.lang_pooler(smiles['attention_mask'], smiles_outputs)


            graph_emb = gcn_output[0]
            token_num = iupac['mlm_input_ids'].shape[1]
            graph_emb = graph_emb.repeat(1, 1, token_num).reshape(graph_emb.shape[0], -1, graph_emb.shape[1])
            
            # graph_emb_concat = torch.zeros_like(lingua_outputs.last_hidden_state)
            # graph_emb_concat[::2] = graph_emb
            # graph_emb_concat[1::2] = graph_emb
            mlm_labels = iupac['mlm_labels']
            features = torch.cat((iupac_outputs.last_hidden_state, graph_emb), dim=2)
            prediction_scores = self.iupac_head(features)
            mlm_loss_lang_iupac = self.loss_mlm(prediction_scores.view(-1, self.iupac_config.vocab_size), mlm_labels.view(-1))
            return_dict['mlm_loss_lang_iupac'] = mlm_loss_lang_iupac


            mlm_labels = smiles['mlm_labels']
            features = torch.cat((smiles_outputs.last_hidden_state, graph_emb), dim=2)
            prediction_scores = self.smiles_head(features)
            mlm_loss_lang_smiles = self.loss_mlm(prediction_scores.view(-1, self.smiles_config.vocab_size), mlm_labels.view(-1))
            return_dict['mlm_loss_lang_smiles'] = mlm_loss_lang_smiles
        
        else:
            # print("contrastive learning")
            iupac_outputs = self.iupac_roberta(
                iupac['input_ids'],
                attention_mask=iupac['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.iupac_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            iupac_pooler_output = self.lang_pooler(iupac['attention_mask'], iupac_outputs)

            smiles_outputs = self.smiles_roberta(
                smiles['input_ids'],
                attention_mask=smiles['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.smiles_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            smiles_pooler_output = self.lang_pooler(smiles['attention_mask'], smiles_outputs)



            gcn_output = self.gnn(graph)
            graph_emb = gcn_output[0]
            
            iupac_project = self.iupac_mlp(iupac_pooler_output)
            smiles_project = self.smiles_mlp(smiles_pooler_output)
            graph_project = self.gcn_mlp(graph_emb)
            

            iupac_emb_norm = nn.functional.normalize(iupac_project, dim=1)
            smiles_emb_norm = nn.functional.normalize(smiles_project, dim=1)            
            graph_emb_norm = nn.functional.normalize(graph_project, dim=1)
            # smiles_emb_norm = lang_emb_norm[::2]
            # iupac_emb_norm = lang_emb_norm[1::2]

            def get_ctr_loss(feat1, feat2, feat2_queue):
                l_pos = torch.einsum('nc,nc->n', [feat1, feat2]).unsqueeze(-1)                
                l_neg = torch.einsum('nc,ck->nk', [feat1, feat2_queue.clone().detach()])
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= self.T
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                loss_ctr = self.loss_mlm(logits, labels)
                return loss_ctr
            
            return_dict['iupac_graph_ctr'] = get_ctr_loss(iupac_emb_norm, graph_emb_norm, self.gcn_queue)
            return_dict['iupac_smiles_ctr'] = get_ctr_loss(iupac_emb_norm, smiles_emb_norm, self.smiles_queue)

            return_dict['smiles_graph_ctr'] = get_ctr_loss(smiles_emb_norm, graph_emb_norm, self.gcn_queue)
            return_dict['smiles_iupac_ctr'] = get_ctr_loss(smiles_emb_norm, iupac_emb_norm, self.iupac_queue)

            return_dict['graph_iupac_ctr'] = get_ctr_loss(graph_emb_norm, iupac_emb_norm, self.iupac_queue)
            return_dict['graph_smiles_ctr'] = get_ctr_loss(graph_emb_norm, smiles_emb_norm, self.smiles_queue)
            
            
            
            self._dequeue_and_enqueue(iupac_emb_norm, self.iupac_queue, self.iupac_queue_ptr)
            self._dequeue_and_enqueue(smiles_emb_norm, self.smiles_queue, self.smiles_queue_ptr)
            self._dequeue_and_enqueue(graph_emb_norm, self.gcn_queue, self.gcn_queue_ptr)
            
            
        loss = 0
        for _, item_loss in return_dict.items():
            loss += item_loss
        if 'iupac_graph_ctr' in return_dict:
            loss = loss / 6
        return_dict['loss'] = loss
        return return_dict


class MultilingualModelConcat(RobertaPreTrainedModel):
    def __init__(self, multilingua_config, gcn_config, atom_vocab_size, K=65536, T=0.07):
        super(MultilingualModelConcat, self).__init__(multilingua_config)
        
        
        # multilingua config vocab_size = (smiles vocab size + iupac vocab size)
        self.multilingua_config = multilingua_config
        self.lang_roberta = RobertaModel(multilingua_config, add_pooling_layer=True)
        self.gnn = DeeperGCN(gcn_config)
        
        self.lm_head = RobertaLMHead(multilingua_config, graph_hidden_size=gcn_config['gnn_embed_dim'] * 2)
        self.atom_head = AtomHead(multilingua_config.hidden_size + gcn_config['gnn_embed_dim'], 
                                  atom_vocab_size, 'gelu', norm=gcn_config['gnn_norm'])
        self.lang_pooler = Pooler(multilingua_config.pooler_type)
        
        self.config = multilingua_config
        self.gcn_config = gcn_config
        self.loss_mlm = nn.CrossEntropyLoss()
        
        
        # corss modal encoder
        # self.cross_encoder = nn.ModuleList([RobertaLayer(multilingua_config) for _ in range(multilingua_config.num_hidden_layers // 4)])
        # cl_init_2(self, iupac_config)
        hidden_size = multilingua_config.hidden_size
        
        # head for the lang
        self.lang_mlp = MLPLayerMoco(multilingua_config)
        # head for the gcn
        self.gcn_mlp = MLPLayerMoco(multilingua_config)
        # gcn embedding converter
        self.gcn_embedding = nn.Linear(gcn_config['gnn_embed_dim'], multilingua_config.hidden_size, bias=True)
        self.LayerNorm = torch.nn.LayerNorm(multilingua_config.hidden_size, eps=1e-12)
        
        
        # memory bank for the lang
        self.register_buffer("lang_queue", torch.randn(hidden_size, K))
        self.lang_queue = nn.functional.normalize(self.lang_queue, dim=0)
        self.register_buffer("lang_queue_ptr", torch.zeros(1, dtype=torch.long))

        # memory bank for the gnn
        self.register_buffer("gcn_queue", torch.randn(hidden_size, K))
        self.gcn_queue = nn.functional.normalize(self.gcn_queue, dim=0)
        self.register_buffer("gcn_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.T = T
        self.K = K
        
    
    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output            
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        # gather keys before updating queue
        keys = self.concat_all_gather(keys)
        # print('keys shape: {}'.format(keys.shape[0]))
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        queue_ptr[0] = ptr
    
    
    
    def forward(self, lingua=None, graph=None, gm_masked_labels=None):
        return_dict = {}
        if lingua['input_ids'].device != self.device:
            lingua['input_ids'] = lingua['input_ids'].to(self.device)
            if 'mlm_input_ids' in lingua:
                lingua['mlm_input_ids'] = lingua['mlm_input_ids'].to(self.device)
                lingua['mlm_labels'] = lingua['mlm_labels'].to(self.device)
            lingua['attention_mask'] = lingua['attention_mask'].to(self.device)
        graph.to(self.device)
        
        if gm_masked_labels is not None:
            # print("mask graph")
            lingua_outputs = self.lang_roberta(
                lingua['input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            lang_pooler_output = self.lang_pooler(lingua['attention_mask'], lingua_outputs)
            gcn_output = self.gnn(graph)
            # split the lang_pooler_output into smiles and iupacs
            # smiles_cls = lang_pooler_output[::2]
            # iupac_cls = lang_pooler_output[1::2]
            graph_batch = graph.batch
            
            graph_lang_logits = self.atom_head(node_features=gcn_output[1], cls_features=lang_pooler_output[graph_batch])
            
            # graph_smiles_logits = self.atom_head(node_features=gcn_output[1], cls_features=smiles_cls[graph_batch])
            # graph_iupac_logits = self.atom_head(node_features=gcn_output[1], cls_features=iupac_cls[graph_batch])
            # graph_smiles_mask_loss = self.loss_mlm(graph_smiles_logits, gm_masked_labels)
            # graph_iupac_mask_loss = self.loss_mlm(graph_iupac_logits, gm_masked_labels)
            
            graph_lang_mask_loss = self.loss_mlm(graph_lang_logits, gm_masked_labels)
            return_dict['graph_lang_mask_loss'] = graph_lang_mask_loss
            # return_dict['graph_smiles_mask_loss'] = graph_smiles_mask_loss
            # return_dict['graph_iupac_mask_loss'] = graph_iupac_mask_loss
        
        elif 'mlm_input_ids' in lingua: # lang mask resume with graph assist
            # print("mask lang")
            gcn_output = self.gnn(graph)
            lingua_outputs = self.lang_roberta(
                lingua['mlm_input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            lang_pooler_output = self.lang_pooler(lingua['attention_mask'], lingua_outputs)
            graph_emb = gcn_output[0]
            token_num = lingua['mlm_input_ids'].shape[1]
            graph_emb = graph_emb.repeat(1, 1, token_num).reshape(graph_emb.shape[0], -1, graph_emb.shape[1])
            mlm_labels = lingua['mlm_labels']
            # graph_emb_concat = torch.zeros_like(lingua_outputs.last_hidden_state)
            # graph_emb_concat[::2] = graph_emb
            # graph_emb_concat[1::2] = graph_emb
            # features = torch.cat((lingua_outputs.last_hidden_state, graph_emb_concat), dim=2)
            
            features = torch.cat((lingua_outputs.last_hidden_state, graph_emb), dim=2)
            prediction_scores = self.lm_head(features)
            mlm_loss_lang = self.loss_mlm(prediction_scores.view(-1, self.multilingua_config.vocab_size), mlm_labels.view(-1))
            return_dict['mlm_loss_lang'] = mlm_loss_lang
        
        else:
            # print("contrastive learning")
            lingua_outputs = self.lang_roberta(
                lingua['input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                )
            lang_pooler_output = self.lang_pooler(lingua['attention_mask'], lingua_outputs)
            gcn_output = self.gnn(graph)
            graph_emb = gcn_output[0]
            # with torch.no_grad():
            
            lang_project = self.lang_mlp(lang_pooler_output)
            graph_project = self.gcn_mlp(graph_emb)
            
            lang_emb_norm = nn.functional.normalize(lang_project, dim=1)  
            graph_emb_norm = nn.functional.normalize(graph_project, dim=1)
            # smiles_emb_norm = lang_emb_norm[::2]
            # iupac_emb_norm = lang_emb_norm[1::2]
            # graph_emb_concat = torch.zeros_like(lang_emb_norm)
            # graph_emb_concat[::2] = graph_emb_norm
            # graph_emb_concat[1::2] = graph_emb_norm
            
            # l_pos = torch.einsum('nc,nc->n', [lang_emb_norm, graph_emb_concat]).unsqueeze(-1)
            
            l_pos = torch.einsum('nc,nc->n', [lang_emb_norm, graph_emb_norm]).unsqueeze(-1)
            l_neg_lang_graph = torch.einsum('nc,ck->nk', [lang_emb_norm, self.gcn_queue.clone().detach()])
            # l_neg_graph_lang = torch.einsum('nc,ck->nk', [graph_emb_concat, self.lang_queue.clone().detach()])
            l_neg_graph_lang = torch.einsum('nc,ck->nk', [graph_emb_norm, self.lang_queue.clone().detach()])
            
            lang_logits = torch.cat([l_pos, l_neg_lang_graph], dim=1)
            graph_logits = torch.cat([l_pos, l_neg_graph_lang], dim=1)
            
            lang_logits /= self.T
            graph_logits /= self.T
            labels = torch.zeros(lang_logits.shape[0], dtype=torch.long).cuda()
            
            
            
            
            # self._dequeue_and_enqueue(graph_emb_concat, self.gcn_queue, self.gcn_queue_ptr)
            self._dequeue_and_enqueue(graph_emb_norm, self.gcn_queue, self.gcn_queue_ptr)
            self._dequeue_and_enqueue(lang_emb_norm, self.lang_queue, self.lang_queue_ptr)
            
            loss_lang_ctr = self.loss_mlm(lang_logits, labels)
            loss_graph_ctr = self.loss_mlm(graph_logits, labels)
            return_dict['loss_lang_ctr'] = loss_lang_ctr
            return_dict['loss_graph_ctr'] = loss_graph_ctr
            
        loss = 0
        for _, item_loss in return_dict.items():
            loss += item_loss
        return_dict['loss'] = loss
        return return_dict




class RobertaHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, regression=False):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        output_dim = config.task_output_dim
        self.out_proj = nn.Linear(config.hidden_size, output_dim)
        self.regression = regression

    def forward(self, features, only_feat=False):
        x = self.dropout(features)
        x = self.dense(x)
        if self.regression:
            x = torch.relu(x)
        else:
            x = torch.tanh(x)
        # try diff norm: batch norm, layernorm
        if only_feat:
            return x
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


from multimodal.modeling_roberta import RobertaModel, RobertaLayer

class MultilingualModelUNI(RobertaPreTrainedModel):
    
    # temp parameter: temperature for the constastive loss
    def __init__(self, multilingua_config, gcn_config, atom_vocab_size, fg_labels_num=85, fingerprint_len=2048, temp=0.05, check_frag=False):
        super(MultilingualModelUNI, self).__init__(multilingua_config)
        
        self.multilingua_config = multilingua_config
        self.lang_roberta = RobertaModel(multilingua_config, add_pooling_layer=True)
        self.gnn = DeeperGCN(gcn_config)
        
        # mask prediction for the lang model
        self.lm_head = RobertaLMHead(multilingua_config)
        self.atom_lm_head = RobertaLMHead(multilingua_config, vocab_size_sp=atom_vocab_size)

        self.lang_vocab_size = multilingua_config.vocab_size
        self.gcn_vocab_size = atom_vocab_size

        # atom mask predcition for the gnn, maybe another RobertaLMHead???
        # self.atom_head = AtomHead(multilingua_config.hidden_size, 
        #                           atom_vocab_size, 'gelu', norm=gcn_config['gnn_norm'])
        
        # todo 1.head for fingerprint regression 2. head function group prediction 3. head for pair matching
        
        self.lang_pooler = Pooler(multilingua_config.pooler_type)
        
        self.config = multilingua_config
        self.gcn_config = gcn_config
        self.loss_mlm = nn.CrossEntropyLoss()
        self.loss_mlm2 = nn.CrossEntropyLoss(reduce=False)
        

        # transfer from gcn embeddings to lang shape
        self.gcn_embedding = nn.Linear(self.gcn_config['gnn_embed_dim'], self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(multilingua_config.hidden_dropout_prob)
        self.LayerNorm = torch.nn.LayerNorm(multilingua_config.hidden_size, eps=1e-12)

        # contrastive head:
        # if smiles_graph: 0, 1; smiles_iupac_graph: 0, 1, 2; 0 means pair match
        contrastive_class_num = multilingua_config.contrastive_class_num
        multilingua_config.task_output_dim = contrastive_class_num
        self.contrastive_head = RobertaHead(multilingua_config)
        self.contrastive_loss = nn.CrossEntropyLoss()
        # self.contrastive_head = nn.Linear(multilingua_config.hidden_size, contrastive_classs_num)
        # function group:
        multilingua_config.task_output_dim = fg_labels_num
        self.fg_task_head = RobertaHead(multilingua_config)
        self.fg_task_loss = nn.BCEWithLogitsLoss(reduction="mean")

        # fingerprint regeression
        multilingua_config.task_output_dim = fingerprint_len
        self.fingerprint_head = RobertaHead(multilingua_config, regression=True)
        self.fingerprint_loss = nn.MSELoss()
        # self.fingerprint_loss = nn.SmoothL1Loss()
        
        # for output token group alignment
        self.lang_group_layer = RobertaLayer(multilingua_config)
        self.sim = Similarity(temp=temp)
        self.ctr_loss = nn.CrossEntropyLoss()

        self.check_frag = check_frag
        
    
    
    
    def forward(self, lingua=None, graph=None, gm_masked_labels=None, fingerprint_labels=None, contrastive_labels=None, function_group_labels=None, graph_labels=None, token_labels=None, graph_attention_mask=None, smiles_lst=None):
        return_dict = {}
        if lingua['input_ids'].device != self.device:
            lingua['input_ids'] = lingua['input_ids'].to(self.device)
            if 'mlm_input_ids' in lingua:
                lingua['mlm_input_ids'] = lingua['mlm_input_ids'].to(self.device)
                lingua['mlm_labels'] = lingua['mlm_labels'].to(self.device)
            lingua['attention_mask'] = lingua['attention_mask'].to(self.device)

        graph.to(self.device)
        
        
        gcn_output = self.gnn(graph)
        batch_size = lingua['input_ids'].shape[0]
        batch_idx = graph.batch
        
        
        # check the symmetric fragment
        if graph_labels is not None and self.check_frag:
            gcn_raw_embeddings = gcn_output[1].detach().cpu()
            for bs in range(batch_size):
                graph_emb = gcn_raw_embeddings[batch_idx == bs]
                graph_labels_sample = torch.tensor(graph_labels[bs])
                max_group_num = max(graph_labels_sample) + 1
                graph_group_emb_set = set([])
                for g_ind in range(max_group_num):
                    graph_group_embedding = graph_emb[graph_labels_sample == g_ind]
                    if graph_group_embedding.nelement():
                        graph_aggregation_emb = torch.mean(graph_group_embedding, dim=0)
                        graph_aggregation_emb_round = np.round(graph_aggregation_emb.numpy(), 2)
                        graph_aggregation_emb_repr = repr(graph_aggregation_emb_round)
                        # if smiles_lst[bs] == 'OC(c1ccccc1)(c1ccccc1)C1CCNC1':
                        #     print('xxxx')
                        # if smiles_lst[bs] == 'COC(=O)C(=O)Cc1ccccc1.O=C(O)C(=O)Cc1ccccc1':
                        #     print('yyyy')
                        if graph_aggregation_emb_repr in graph_group_emb_set:
                            graph_labels_sample[graph_labels_sample == g_ind] = -1
                            # print(f'-----------same frag occur------------- {smiles_lst[bs]}')
                        else:
                            graph_group_emb_set.add(graph_aggregation_emb_repr)
                graph_labels[bs] = graph_labels_sample.tolist()

        # concat graph atome embeddings and langua embeddings
        gcn_embedding_output = self.gcn_embedding(gcn_output[1])
        gcn_embedding_output = self.LayerNorm(gcn_embedding_output)
        gcn_embedding_output = self.dropout(gcn_embedding_output)


        # pad the gcn_embedding same shape with pos_coord_matrix_pad
        gcn_embedding_lst = []
        for bs in range(batch_size):
            gcn_embedding_lst.append(gcn_embedding_output[batch_idx == bs])
        
        if 'mlm_input_ids' in lingua:
            lang_gcn_outputs, lang_gcn_attention_mask = self.lang_roberta(
                lingua['mlm_input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,

                graph_input = gcn_embedding_lst,
                graph_batch = graph.batch,
                # graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
                gnn_mask_labels = gm_masked_labels,
                graph_attention_mask = graph_attention_mask,
                )
            lang_token_size = lingua['input_ids'].size(1)
            lang_token_embedding = lang_gcn_outputs.last_hidden_state[:, :lang_token_size, :]
            gcn_token_embedding = lang_gcn_outputs.last_hidden_state[:, lang_token_size:, :]
            lang_prediction_scores = self.lm_head(lang_token_embedding)
            mlm_labels = lingua['mlm_labels']

            # mlm_loss_lang = self.loss_mlm(lang_prediction_scores.view(-1, self.lang_vocab_size), mlm_labels.view(-1))

            mlm_loss_lang_2 = self.loss_mlm2(lang_prediction_scores.view(-1, self.lang_vocab_size), mlm_labels.view(-1))
            h_loss_idx = mlm_loss_lang_2.size(0) // 2
            h_batch_size = mlm_labels.size(0) // 2
            
            mlm_loss_lang_token_level = mlm_loss_lang_2[:h_loss_idx].sum() / (mlm_labels[:h_batch_size, :] != -100).sum()
            mlm_loss_lang_frag_level = mlm_loss_lang_2[h_loss_idx:].sum() / (mlm_labels[h_batch_size:, :] != -100).sum()

            return_dict['mlm_loss_lang_token_level'] = mlm_loss_lang_token_level
            return_dict['mlm_loss_lang_frag_level'] = mlm_loss_lang_frag_level

            gcn_prediction_scores = self.atom_lm_head(gcn_token_embedding)
            # mlm_loss_gcn = self.loss_mlm(gcn_prediction_scores.view(-1, self.gcn_vocab_size), gm_masked_labels.view(-1))

            mlm_loss_gcn_2 = self.loss_mlm2(gcn_prediction_scores.view(-1, self.gcn_vocab_size), gm_masked_labels.view(-1))
            h_loss_idx = mlm_loss_gcn_2.size(0) // 2
            h_batch_size = gm_masked_labels.size(0) // 2
            
            mlm_loss_gcn_token_level = mlm_loss_gcn_2[:h_loss_idx].sum() / (gm_masked_labels[:h_batch_size, :] != -100).sum()
            mlm_loss_gcn_frag_level = mlm_loss_gcn_2[h_loss_idx:].sum() / (gm_masked_labels[h_batch_size:, :] != -100).sum()

            return_dict['mlm_loss_gcn_token_level'] = mlm_loss_gcn_token_level
            return_dict['mlm_loss_gcn_frag_level'] = mlm_loss_gcn_frag_level

        
        else:
            # group level alignment between graph and language model
            lang_gcn_outputs, lang_gcn_attention_mask = self.lang_roberta(
                lingua['input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,

                graph_input = gcn_embedding_lst,
                graph_batch = graph.batch,
                # graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
                gnn_mask_labels = None,
                graph_attention_mask = graph_attention_mask,
                )
            
            batch_size = lingua['input_ids'].shape[0]
            pos_num = (contrastive_labels == 0).sum().item()
            if pos_num == batch_size: # all postive pair: group alignment
                # split the graph and lang embeddings
                last_hidden_embedding = lang_gcn_outputs['last_hidden_state']
                graph_batch = graph.batch
                lang_input_dim = lingua['input_ids'].shape[1]
                last_lang_embedding = last_hidden_embedding[:, :lang_input_dim]
                last_graph_embedding = last_hidden_embedding[:, lang_input_dim:]
                max_atom_num = last_graph_embedding.shape[1]
                
                
                hidden_size = last_hidden_embedding.shape[-1]
                
                
                group_align_loss = 0
                
                
                # for the whole batch
                all_graph_group_emb_lst = []
                all_lang_group_emb_lst = []
                
                for i in range(batch_size):
                    lang_token_label = token_labels[i]
                    lang_emb = last_lang_embedding[i]
                    
                    
                    graph_label = graph_labels[i]
                    # truncation
                    graph_label = graph_label[:max_atom_num]
                    graph_label = torch.tensor(graph_label, device=self.device)
                    atom_num = (graph_batch == i).sum().item()
                    trunc_atom_num = min(atom_num, max_atom_num)
                    assert trunc_atom_num == len(graph_label)
                    graph_emb = last_graph_embedding[i][:trunc_atom_num]
                    
                    group_num = max(graph_labels[i]) + 1
                    
                    
                    lang_group_emb_lst = []
                    max_lang_group_size = 0
                    
                    
                    valid_cnt = 0
                    for j in range(group_num):
                        # language may miss some group nums
                        lang_group_embedding = lang_emb[lang_token_label == j]
                        if lang_group_embedding.nelement():
                            graph_group_embedding = graph_emb[graph_label == j]
                            if graph_group_embedding.nelement():
                                # both have embeddings, begin the alignment
                                graph_aggregation_emb = torch.mean(graph_group_embedding, dim=0)
                                all_graph_group_emb_lst.append(graph_aggregation_emb)
                                
                                lang_group_emb_lst.append(lang_group_embedding)
                                lang_group_size = lang_group_embedding.shape[0]
                                if lang_group_size > max_lang_group_size:
                                    max_lang_group_size = lang_group_size
                                # for lang, avg
                                # lang_aggregation_emb = self.lang_group_layer(lang_group_embedding)
                                # lang_aggregation_emb = lang_aggregation_emb.sum(dim=1) / lang_aggregation_emb.shape[1]
                                
                                valid_cnt += 1
                                
                            else:
                                pass
                                # print("xxxx")
                    
                    # assemble the embedding
                    assemb_lang_group_embedding = torch.full((valid_cnt, max_lang_group_size, hidden_size), 0, \
                        dtype=lang_emb.dtype, device=self.device)
                    assemb_lang_group_mask = torch.full((valid_cnt, max_lang_group_size), 0, \
                        dtype=lang_emb.dtype, device=self.device)
                    
                    for k, ele in enumerate(lang_group_emb_lst):
                        ele_len = ele.shape[0]
                        assemb_lang_group_embedding[k, :ele_len] = ele
                        assemb_lang_group_mask[k, :ele_len] = 1
                    
                    # change the assemb_lang_group_mask to attention mask
                    assemb_atten_mask = self.lang_roberta.get_extended_attention_mask(assemb_lang_group_mask, \
                        assemb_lang_group_mask.size(), self.device)
                    
                    assemb_group_output = self.lang_group_layer(assemb_lang_group_embedding, attention_mask=assemb_atten_mask)
                    
                    
                    assemb_group_pool_output = (assemb_group_output[0] * assemb_lang_group_mask.unsqueeze(-1)).sum(1) / assemb_lang_group_mask.sum(-1).unsqueeze(-1)
                    # assemb_group_pool_output = self.lang_pooler(assemb_lang_group_mask, assemb_group_output[0])

                    
                    all_lang_group_emb_lst.append(assemb_group_pool_output)
                    # calculate the cos distance
                    # assemb_group_graph_output = torch.stack(graph_group_emb_lst)
                    
                    
                    
                    
                    # def cos_distance(m1, m2):
                    #     mn1 = F.normalize(m1, p=2, dim=1)
                    #     mn2 = F.normalize(m2, p=2, dim=1)
                    #     cos_dis = (mn1 - mn2) ** 2
                    #     cos_dis = cos_dis.sum(dim=1) / 2
                    #     return torch.mean(cos_dis)
                    # cos_dis = cos_distance(assemb_group_pool_output, assemb_group_graph_output)
                    # group_align_loss += cos_dis
                
                
                graph_emb_all = torch.stack(all_graph_group_emb_lst)
                lang_emb_all = torch.cat(all_lang_group_emb_lst, dim=0)
                assert graph_emb_all.shape[0] == lang_emb_all.shape[0]
                
                cos_sim = self.sim(graph_emb_all.unsqueeze(1), lang_emb_all.unsqueeze(0))
                ctr_sample_num = cos_sim.shape[0]
                labels = torch.arange(ctr_sample_num).long().to(self.device)
                
                return_dict['group_align_loss'] = self.ctr_loss(cos_sim, labels) + self.ctr_loss(cos_sim.t(), labels)
            
            
                
        lang_gcn_pooler_output = self.lang_pooler(lang_gcn_attention_mask, lang_gcn_outputs)

        if gm_masked_labels is None: # no SGM on masked sample, little mask may change the molecule
            ctr_logits = self.contrastive_head(lang_gcn_pooler_output)
            ctr_loss = self.contrastive_loss(ctr_logits, contrastive_labels)
            return_dict['ctr_loss'] = ctr_loss
        # only pair-matched loss have fg loss and finger print regression loss, except token-level random mask
        # at least one modality is intact
        fg_mask = (contrastive_labels == 0) # zero means positive pair
        
        if gm_masked_labels is not None: # mask
            h_batch_size = contrastive_labels.size(0) // 2
            fg_mask[:h_batch_size] = False # except the random mask batch, fragment (at least one modality is intact)

        # function group loss
        fg_logits = self.fg_task_head(lang_gcn_pooler_output)
        fg_loss = self.fg_task_loss(fg_logits[fg_mask], function_group_labels[fg_mask])
        return_dict['fg_loss'] = fg_loss

        # finger print regression loss
        finger_print_logits = self.fingerprint_head(lang_gcn_pooler_output)
        finger_print_loss = self.fingerprint_loss(finger_print_logits[fg_mask], fingerprint_labels[fg_mask])
        return_dict['finger_print_loss'] = finger_print_loss

        loss = 0
        for _, item_loss in return_dict.items():
            if torch.isnan(item_loss).sum():
                print("nan")
                import pdb; pdb.set_trace()
            loss += item_loss
        return_dict['loss'] = loss
        return return_dict





# task_type: ctl, cls, reg


class MultilingualModelUNIPLI(RobertaPreTrainedModel):
    
    # temp parameter: temperature for the constastive loss
    def __init__(self, multilingua_config, gcn_config, temp=0.05, protein_emb_size=1280, task_type='ctl', eval_mode=False, extract_feat=False):
        super(MultilingualModelUNIPLI, self).__init__(multilingua_config)
        
        self.multilingua_config = multilingua_config
        self.lang_roberta = RobertaModel(multilingua_config, add_pooling_layer=True)
        self.gnn = DeeperGCN(gcn_config)
        

        
        self.lang_pooler = Pooler(multilingua_config.pooler_type)
        
        self.config = multilingua_config
        self.gcn_config = gcn_config
        self.loss_mlm = nn.CrossEntropyLoss()
        

        
        # transfer from gcn embeddings to lang shape
        self.gcn_embedding = nn.Linear(self.gcn_config['gnn_embed_dim'], self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(multilingua_config.hidden_dropout_prob)
        self.LayerNorm = torch.nn.LayerNorm(multilingua_config.hidden_size, eps=1e-12)


        # head for protein feature converter
        # head for mol feature converter
        tmp_config = copy.deepcopy(multilingua_config)
        org_hidden_size = tmp_config.hidden_size
        tmp_config.task_output_dim = tmp_config.hidden_size
        self.mol_feature_head = RobertaHead(tmp_config)
        

        tmp_config.hidden_size = protein_emb_size
        self.pro_feature_head = RobertaHead(tmp_config)
        
        self.task_type = task_type

        assert self.task_type in ['ctl', 'cls', 'reg']
        # if cls or reg, we add additional head
        if self.task_type != 'ctl':
            # need concat the feature
            self.mln = nn.LayerNorm(org_hidden_size)
            self.pln = nn.LayerNorm(org_hidden_size)
            tmp_config.hidden_size = org_hidden_size * 2
            if self.task_type == 'cls':
                tmp_config.task_output_dim = 2
                self.pli_head = RobertaHead(tmp_config)
                self.cls_loss = nn.CrossEntropyLoss()
            elif self.task_type == 'reg':
                tmp_config.task_output_dim = 1
                self.pli_head = RobertaHead(tmp_config, regression=True)
                self.reg_loss = nn.SmoothL1Loss()
        else:
            self.sim = Similarity(temp=temp)
            self.ctl_loss = nn.CrossEntropyLoss()

        self.eval_mode = eval_mode
        self.extract_feat = extract_feat


        # # contrastive head:
        # # if smiles_graph: 0, 1; smiles_iupac_graph: 0, 1, 2; 0 means pair match
        # contrastive_class_num = multilingua_config.contrastive_class_num
        # multilingua_config.task_output_dim = contrastive_class_num
        # self.contrastive_head = RobertaHead(multilingua_config)
        # self.contrastive_loss = nn.CrossEntropyLoss()
        # # self.contrastive_head = nn.Linear(multilingua_config.hidden_size, contrastive_classs_num)
        # # function group:
        # multilingua_config.task_output_dim = fg_labels_num
        # self.fg_task_head = RobertaHead(multilingua_config)
        # self.fg_task_loss = nn.BCEWithLogitsLoss(reduction="mean")

        # # fingerprint regeression
        # multilingua_config.task_output_dim = fingerprint_len
        # self.fingerprint_head = RobertaHead(multilingua_config, regression=True)
        # self.fingerprint_loss = nn.MSELoss()
        # # self.fingerprint_loss = nn.SmoothL1Loss()
        
        # # for output token group alignment
        # self.lang_group_layer = RobertaLayer(multilingua_config)
        # self.sim = Similarity(temp=temp)
        # self.ctr_loss = nn.CrossEntropyLoss()
        
    
    
    
    def forward(self, lingua=None, graph=None, pli_labels=None,
    protein_embedding=None):

        if protein_embedding is not None:
            # transform protein features
            protein_embedding = protein_embedding.to(self.device)
            protein_embedding = self.pro_feature_head(protein_embedding)
            if lingua is None: # eval model, extract protein feat only
                return protein_embedding

        return_dict = {}
        if lingua['input_ids'].device != self.device:
            lingua['input_ids'] = lingua['input_ids'].to(self.device)
            if 'mlm_input_ids' in lingua:
                lingua['mlm_input_ids'] = lingua['mlm_input_ids'].to(self.device)
                lingua['mlm_labels'] = lingua['mlm_labels'].to(self.device)
            lingua['attention_mask'] = lingua['attention_mask'].to(self.device)


        # change protein_embedding
        graph.to(self.device)
        
        
        gcn_output = self.gnn(graph)
        # concat graph atome embeddings and langua embeddings
        gcn_embedding_output = self.gcn_embedding(gcn_output[1])
        gcn_embedding_output = self.LayerNorm(gcn_embedding_output)
        gcn_embedding_output = self.dropout(gcn_embedding_output)

        lang_gcn_outputs, lang_gcn_attention_mask, _ = self.lang_roberta(
                lingua['input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,

                graph_input = gcn_embedding_output,
                graph_batch = graph.batch,
                graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
                gnn_mask_labels = None
                )
        lang_gcn_pooler_output = self.lang_pooler(lang_gcn_attention_mask, lang_gcn_outputs)
        # transform mol features
        mol_embedding = self.mol_feature_head(lang_gcn_pooler_output)

        if protein_embedding is None:
            return mol_embedding
        

        # check if eval model, if eval return score
        if self.eval_mode:
            if self.task_type == 'ctl':
                if self.extract_feat:
                    return_dict = {}
                    return_dict['protein_emb'] = protein_embedding
                    return_dict['mol_emb'] = mol_embedding
                    return return_dict
                else:
                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    cosine_simi = cos(protein_embedding, mol_embedding)
                    return cosine_simi
            else:
                mol_embedding = self.mln(mol_embedding)
                protein_embedding = self.pln(protein_embedding)
                c_featrue = torch.cat([mol_embedding, protein_embedding], dim=1)
                logits = self.pli_head(c_featrue)
                return logits
                # pass        
         
        
        if self.task_type == 'ctl':
            # labels should all 1
            return_dict = {}

            assert torch.all(pli_labels==1)
            cos_sim = self.sim(protein_embedding.unsqueeze(1), mol_embedding.unsqueeze(0))
            ctr_sample_num = cos_sim.shape[0]
            labels = torch.arange(ctr_sample_num).long().to(self.device)
            
            return_dict['ctl_loss'] = self.ctl_loss(cos_sim, labels) + self.ctl_loss(cos_sim.t(), labels)
        else:
            # concat feature
            mol_embedding = self.mln(mol_embedding)
            protein_embedding = self.pln(protein_embedding)
            c_featrue = torch.cat([mol_embedding, protein_embedding], dim=1)
            logits = self.pli_head(c_featrue)
            if self.task_type == 'cls':
                return_dict['cls_loss'] = self.cls_loss(logits, pli_labels)
            else: # regression
                # todo, consider label
                return_dict['reg_loss'] = self.reg_loss(logits, pli_labels)

        loss = 0
        for _, item_loss in return_dict.items():
            if torch.isnan(item_loss).sum():
                print("nan")
            loss += item_loss
        return_dict['loss'] = loss
        return return_dict



class MultilingualModelUNIFormer(RobertaPreTrainedModel):
    
    # temp parameter: temperature for the constastive loss
    def __init__(self, multilingua_config, gcn_config, atom_vocab_size, fg_labels_num=85, fingerprint_len=2048, temp=0.05):
        super(MultilingualModelUNIFormer, self).__init__(multilingua_config)
        
        self.multilingua_config = multilingua_config
        self.lang_roberta = RobertaModel(multilingua_config, add_pooling_layer=True)
        
        # 
        self.gnn = GraphormerEncoder(gcn_config)
        
        # mask prediction for the lang model
        self.lm_head = RobertaLMHead(multilingua_config, atom_vocab_size=atom_vocab_size)

        self.lang_gcn_vocab_size = multilingua_config.vocab_size + atom_vocab_size

        # atom mask predcition for the gnn, maybe another RobertaLMHead???
        # self.atom_head = AtomHead(multilingua_config.hidden_size, 
        #                           atom_vocab_size, 'gelu', norm=gcn_config['gnn_norm'])
        
        # todo 1.head for fingerprint regression 2. head function group prediction 3. head for pair matching
        
        self.lang_pooler = Pooler(multilingua_config.pooler_type)
        
        self.config = multilingua_config
        self.gcn_config = gcn_config
        self.loss_mlm = nn.CrossEntropyLoss()
        

        # transfer from gcn embeddings to lang shape
        # self.gcn_embedding = nn.Linear(self.gcn_config['gnn_embed_dim'], self.config.hidden_size, bias=True)
        # self.dropout = nn.Dropout(multilingua_config.hidden_dropout_prob)
        # self.LayerNorm = torch.nn.LayerNorm(multilingua_config.hidden_size, eps=1e-12)

        # contrastive head:
        # if smiles_graph: 0, 1; smiles_iupac_graph: 0, 1, 2; 0 means pair match
        contrastive_class_num = multilingua_config.contrastive_class_num
        multilingua_config.task_output_dim = contrastive_class_num
        self.contrastive_head = RobertaHead(multilingua_config)
        self.contrastive_loss = nn.CrossEntropyLoss()
        # self.contrastive_head = nn.Linear(multilingua_config.hidden_size, contrastive_classs_num)
        # function group:
        multilingua_config.task_output_dim = fg_labels_num
        self.fg_task_head = RobertaHead(multilingua_config)
        self.fg_task_loss = nn.BCEWithLogitsLoss(reduction="mean")

        # fingerprint regeression
        multilingua_config.task_output_dim = fingerprint_len
        self.fingerprint_head = RobertaHead(multilingua_config, regression=True)
        self.fingerprint_loss = nn.MSELoss()
        # self.fingerprint_loss = nn.SmoothL1Loss()
        
        # for output token group alignment
        self.lang_group_layer = RobertaLayer(multilingua_config)
        self.sim = Similarity(temp=temp)
        self.ctr_loss = nn.CrossEntropyLoss()
        
    
    
    
    def forward(self, lingua=None, graph=None, gm_masked_labels=None, fingerprint_labels=None, contrastive_labels=None, function_group_labels=None, graph_labels=None, token_labels=None, graph_batch=None):
        return_dict = {}
        if lingua['input_ids'].device != self.device:
            lingua['input_ids'] = lingua['input_ids'].to(self.device)
            if 'mlm_input_ids' in lingua:
                lingua['mlm_input_ids'] = lingua['mlm_input_ids'].to(self.device)
                lingua['mlm_labels'] = lingua['mlm_labels'].to(self.device)
            lingua['attention_mask'] = lingua['attention_mask'].to(self.device)

        graph_batch.to(self.device)
        
        
        # gcn_output = self.gnn(graph)
        # # concat graph atome embeddings and langua embeddings
        # gcn_embedding_output = self.gcn_embedding(gcn_output[1])
        # gcn_embedding_output = self.LayerNorm(gcn_embedding_output)
        # gcn_embedding_output = self.dropout(gcn_embedding_output)
        gcn_embedding_output = self.gnn(graph) # gcn_output: batch_size x max_node x graph_dim
        

        # get mask and concat
        data_x = graph["x"]
        n_graph, n_node = data_x.size()[:2]
        padding_mask = ~((data_x[:, :, 0]).eq(0))  # B x T x 1
        
        emb_dim = gcn_embedding_output.shape[-1]
        gcn_embedding_output = gcn_embedding_output[:, 1:, :].reshape(-1, emb_dim) # erase cls embedding and reshape to 
        # (batch_size x max_node) x feat_dim
        padding_mask = padding_mask.reshape(gcn_embedding_output.shape[0])
        gcn_embedding_output = gcn_embedding_output[padding_mask]

        assert gcn_embedding_output.shape[0] == graph_batch.shape[0]

        # todo need change max graph size
        if 'mlm_input_ids' in lingua:
            lang_gcn_outputs, lang_gcn_attention_mask, graph_mm_labels = self.lang_roberta(
                lingua['mlm_input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,

                graph_input = gcn_embedding_output,
                graph_batch = graph_batch,
                graph_max_seq_size = self.gcn_config['max_nodes'],
                gnn_mask_labels = gm_masked_labels
                )
            lang_gcn_pooler_output = self.lang_pooler(lang_gcn_attention_mask, lang_gcn_outputs)
            # graph_emb = gcn_output[0]
            prediction_scores = self.lm_head(lang_gcn_outputs.last_hidden_state)
            mlm_labels = lingua['mlm_labels']
            # concat mlm_labels and graph_mm_labels
            lang_gcn_mlm_labels = torch.cat((mlm_labels, graph_mm_labels), dim=1)

            mlm_loss_lang_gcn = self.loss_mlm(prediction_scores.view(-1, self.lang_gcn_vocab_size), lang_gcn_mlm_labels.view(-1))
            
            if lang_gcn_mlm_labels.max() > 4500:
                print("max label biger than 4500!!")
            
            return_dict['mlm_loss_lang_gcn'] = mlm_loss_lang_gcn

        
        else:
            # group level alignment between graph and language model
            lang_gcn_outputs, lang_gcn_attention_mask, _ = self.lang_roberta(
                lingua['input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,

                graph_input = gcn_embedding_output,
                graph_batch = graph_batch,
                graph_max_seq_size = self.gcn_config['max_nodes'],
                gnn_mask_labels = None
                )
            
            # split the graph and lang embeddings
            last_hidden_embedding = lang_gcn_outputs['last_hidden_state']
            # graph_batch = graph.batch
            lang_input_dim = lingua['input_ids'].shape[1]
            last_lang_embedding = last_hidden_embedding[:, :lang_input_dim]
            last_graph_embedding = last_hidden_embedding[:, lang_input_dim:]
            max_atom_num = last_graph_embedding.shape[1]
            
            batch_size = lingua['input_ids'].shape[0]
            hidden_size = last_hidden_embedding.shape[-1]
            
            
            group_align_loss = 0
            
            
            # for the whole batch
            all_graph_group_emb_lst = []
            all_lang_group_emb_lst = []
            
            for i in range(batch_size):
                lang_token_label = token_labels[i]
                lang_emb = last_lang_embedding[i]
                
                
                graph_label = graph_labels[i]
                # truncation
                graph_label = graph_label[:max_atom_num]
                graph_label = torch.tensor(graph_label, device=self.device)
                atom_num = (graph_batch == i).sum().item()
                trunc_atom_num = min(atom_num, max_atom_num)
                assert trunc_atom_num == len(graph_label)
                graph_emb = last_graph_embedding[i][:trunc_atom_num]
                
                group_num = max(graph_labels[i]) + 1
                
                
                lang_group_emb_lst = []
                max_lang_group_size = 0
                
                
                valid_cnt = 0
                for j in range(group_num):
                    # language may miss some group nums
                    lang_group_embedding = lang_emb[lang_token_label == j]
                    if lang_group_embedding.nelement():
                        graph_group_embedding = graph_emb[graph_label == j]
                        if graph_group_embedding.nelement():
                            # both have embeddings, begin the alignment
                            graph_aggregation_emb = torch.mean(graph_group_embedding, dim=0)
                            all_graph_group_emb_lst.append(graph_aggregation_emb)
                            
                            lang_group_emb_lst.append(lang_group_embedding)
                            lang_group_size = lang_group_embedding.shape[0]
                            if lang_group_size > max_lang_group_size:
                                max_lang_group_size = lang_group_size
                            # for lang, avg
                            # lang_aggregation_emb = self.lang_group_layer(lang_group_embedding)
                            # lang_aggregation_emb = lang_aggregation_emb.sum(dim=1) / lang_aggregation_emb.shape[1]
                            
                            valid_cnt += 1
                            
                        else:
                            pass
                            # print("xxxx")
                
                # assemble the embedding
                assemb_lang_group_embedding = torch.full((valid_cnt, max_lang_group_size, hidden_size), 0, \
                    dtype=lang_emb.dtype, device=self.device)
                assemb_lang_group_mask = torch.full((valid_cnt, max_lang_group_size), 0, \
                    dtype=lang_emb.dtype, device=self.device)
                
                for k, ele in enumerate(lang_group_emb_lst):
                    ele_len = ele.shape[0]
                    assemb_lang_group_embedding[k, :ele_len] = ele
                    assemb_lang_group_mask[k, :ele_len] = 1
                
                # change the assemb_lang_group_mask to attention mask
                assemb_atten_mask = self.lang_roberta.get_extended_attention_mask(assemb_lang_group_mask, \
                    assemb_lang_group_mask.size(), self.device)
                
                assemb_group_output = self.lang_group_layer(assemb_lang_group_embedding, attention_mask=assemb_atten_mask)
                
                
                assemb_group_pool_output = (assemb_group_output[0] * assemb_lang_group_mask.unsqueeze(-1)).sum(1) / assemb_lang_group_mask.sum(-1).unsqueeze(-1)
                # assemb_group_pool_output = self.lang_pooler(assemb_lang_group_mask, assemb_group_output[0])

                
                all_lang_group_emb_lst.append(assemb_group_pool_output)
                # calculate the cos distance
                # assemb_group_graph_output = torch.stack(graph_group_emb_lst)
                
                
                
                
                # def cos_distance(m1, m2):
                #     mn1 = F.normalize(m1, p=2, dim=1)
                #     mn2 = F.normalize(m2, p=2, dim=1)
                #     cos_dis = (mn1 - mn2) ** 2
                #     cos_dis = cos_dis.sum(dim=1) / 2
                #     return torch.mean(cos_dis)
                # cos_dis = cos_distance(assemb_group_pool_output, assemb_group_graph_output)
                # group_align_loss += cos_dis
            
            
            graph_emb_all = torch.stack(all_graph_group_emb_lst)
            lang_emb_all = torch.cat(all_lang_group_emb_lst, dim=0)
            assert graph_emb_all.shape[0] == lang_emb_all.shape[0]
            
            cos_sim = self.sim(graph_emb_all.unsqueeze(1), lang_emb_all.unsqueeze(0))
            ctr_sample_num = cos_sim.shape[0]
            labels = torch.arange(ctr_sample_num).long().to(self.device)
            
            return_dict['group_align_loss'] = self.ctr_loss(cos_sim, labels) + self.ctr_loss(cos_sim.t(), labels)
            
            
            # pick rows according to the iupac_fea
            # iupac_rows = cos_sim[:iupac_number, :]
            # # pick rows according to the smiles
            # smiles_rows = cos_sim[:, :iupac_number].t()
            # labels = torch.arange(iupac_number).long().to(cls.device)
            
            # ctr_loss_iupac = cls.loss_mlm(iupac_rows, labels)
            # ctr_loss_smiles = cls.loss_mlm(smiles_rows, labels)
            
            # return_dict['ctr_loss_iupac'] = ctr_loss_iupac
            
            
            
            
            # return_dict['group_align_loss'] = group_align_loss / batch_size
            
                
            lang_gcn_pooler_output = self.lang_pooler(lang_gcn_attention_mask, lang_gcn_outputs)
            pass

        # contrastive loss
        ctr_logits = self.contrastive_head(lang_gcn_pooler_output)
        ctr_loss = self.contrastive_loss(ctr_logits, contrastive_labels)
        return_dict['ctr_loss'] = ctr_loss
        # only pair-matched loss have fg loss and finger print regression loss
        fg_mask = (contrastive_labels == 0)
        
        # function group loss
        fg_logits = self.fg_task_head(lang_gcn_pooler_output)
        fg_loss = self.fg_task_loss(fg_logits[fg_mask], function_group_labels[fg_mask])
        return_dict['fg_loss'] = fg_loss

        # finger print regression loss
        finger_print_logits = self.fingerprint_head(lang_gcn_pooler_output)
        finger_print_loss = self.fingerprint_loss(finger_print_logits[fg_mask], fingerprint_labels[fg_mask])
        return_dict['finger_print_loss'] = finger_print_loss

        loss = 0
        for _, item_loss in return_dict.items():
            if torch.isnan(item_loss).sum():
                print("nan")
            loss += item_loss
        return_dict['loss'] = loss
        return return_dict



class MultilingualModelUNIEVAL(RobertaPreTrainedModel):
    
    # temp parameter: temperature for the constastive loss
    def __init__(self, multilingua_config, gcn_config, atom_vocab_size, fg_labels_num=85, fingerprint_len=2048, temp=0.05):
        super(MultilingualModelUNIEVAL, self).__init__(multilingua_config)
        
        self.multilingua_config = multilingua_config
        self.lang_roberta = RobertaModel(multilingua_config, add_pooling_layer=True)
        self.gnn = DeeperGCN(gcn_config)
        
        # mask prediction for the lang model
        self.lm_head = RobertaLMHead(multilingua_config, atom_vocab_size=atom_vocab_size)

        self.lang_gcn_vocab_size = multilingua_config.vocab_size + atom_vocab_size

        # atom mask predcition for the gnn, maybe another RobertaLMHead???
        # self.atom_head = AtomHead(multilingua_config.hidden_size, 
        #                           atom_vocab_size, 'gelu', norm=gcn_config['gnn_norm'])
        
        # todo 1.head for fingerprint regression 2. head function group prediction 3. head for pair matching
        
        self.lang_pooler = Pooler(multilingua_config.pooler_type)
        
        self.config = multilingua_config
        self.gcn_config = gcn_config
        self.loss_mlm = nn.CrossEntropyLoss()
        

        # transfer from gcn embeddings to lang shape
        self.gcn_embedding = nn.Linear(self.gcn_config['gnn_embed_dim'], self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(multilingua_config.hidden_dropout_prob)
        self.LayerNorm = torch.nn.LayerNorm(multilingua_config.hidden_size, eps=1e-12)

        # contrastive head:
        # if smiles_graph: 0, 1; smiles_iupac_graph: 0, 1, 2; 0 means pair match
        contrastive_class_num = multilingua_config.contrastive_class_num
        multilingua_config.task_output_dim = contrastive_class_num
        self.contrastive_head = RobertaHead(multilingua_config)
        self.contrastive_loss = nn.CrossEntropyLoss()
        # self.contrastive_head = nn.Linear(multilingua_config.hidden_size, contrastive_classs_num)
        # function group:
        multilingua_config.task_output_dim = fg_labels_num
        self.fg_task_head = RobertaHead(multilingua_config)
        self.fg_task_loss = nn.BCEWithLogitsLoss(reduction="mean")

        # fingerprint regeression
        multilingua_config.task_output_dim = fingerprint_len
        self.fingerprint_head = RobertaHead(multilingua_config, regression=True)
        self.fingerprint_loss = nn.MSELoss()
        # self.fingerprint_loss = nn.SmoothL1Loss()
        
        # for output token group alignment
        self.lang_group_layer = RobertaLayer(multilingua_config)
        self.sim = Similarity(temp=temp)
        self.ctr_loss = nn.CrossEntropyLoss()
        
    
    
    
    def forward(self, lingua=None, graph=None, gm_masked_labels=None, fingerprint_labels=None, contrastive_labels=None, function_group_labels=None, graph_labels=None, token_labels=None):
        return_dict = {}
        if lingua['input_ids'].device != self.device:
            lingua['input_ids'] = lingua['input_ids'].to(self.device)
            if 'mlm_input_ids' in lingua:
                lingua['mlm_input_ids'] = lingua['mlm_input_ids'].to(self.device)
                lingua['mlm_labels'] = lingua['mlm_labels'].to(self.device)
            lingua['attention_mask'] = lingua['attention_mask'].to(self.device)

        graph.to(self.device)
        
        
        gcn_output = self.gnn(graph)
        # concat graph atome embeddings and langua embeddings
        gcn_embedding_output = self.gcn_embedding(gcn_output[1])
        gcn_embedding_output = self.LayerNorm(gcn_embedding_output)
        gcn_embedding_output = self.dropout(gcn_embedding_output)

        assert 'mlm_input_ids' not in lingua
        
        
        lang_gcn_outputs, lang_gcn_attention_mask, _ = self.lang_roberta(
                lingua['input_ids'],
                attention_mask=lingua['attention_mask'],
                # token_type_ids=lingua['token_type_ids'],
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,

                graph_input = gcn_embedding_output,
                graph_batch = graph.batch,
                graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
                gnn_mask_labels = None
                )


        last_hidden_embedding = lang_gcn_outputs['last_hidden_state']
        graph_batch = graph.batch
        lang_input_dim = lingua['input_ids'].shape[1]
        last_lang_embedding = last_hidden_embedding[:, :lang_input_dim]
        last_graph_embedding = last_hidden_embedding[:, lang_input_dim:]
        
        
        max_atom_num = last_graph_embedding.shape[1]
            
        batch_size = lingua['input_ids'].shape[0]
        hidden_size = last_hidden_embedding.shape[-1]
        group_align_loss = 0
        all_graph_group_emb_lst = []
        all_lang_group_emb_lst = []
        
        for i in range(batch_size):
            lang_token_label = token_labels[i]
            lang_emb = last_lang_embedding[i]
            
            
            graph_label = graph_labels[i]
            # truncation
            graph_label = graph_label[:max_atom_num]
            graph_label = torch.tensor(graph_label, device=self.device)
            atom_num = (graph_batch == i).sum().item()
            trunc_atom_num = min(atom_num, max_atom_num)
            assert trunc_atom_num == len(graph_label)
            graph_emb = last_graph_embedding[i][:trunc_atom_num]
            
            group_num = max(graph_labels[i]) + 1
            
            
            lang_group_emb_lst = []
            max_lang_group_size = 0
            
            
            valid_cnt = 0
            for j in range(group_num):
                # language may miss some group nums
                lang_group_embedding = lang_emb[lang_token_label == j]
                if lang_group_embedding.nelement():
                    graph_group_embedding = graph_emb[graph_label == j]
                    if graph_group_embedding.nelement():
                        # both have embeddings, begin the alignment
                        graph_aggregation_emb = torch.mean(graph_group_embedding, dim=0)
                        all_graph_group_emb_lst.append(graph_aggregation_emb)
                        
                        lang_group_emb_lst.append(lang_group_embedding)
                        lang_group_size = lang_group_embedding.shape[0]
                        if lang_group_size > max_lang_group_size:
                            max_lang_group_size = lang_group_size
                        # for lang, avg
                        # lang_aggregation_emb = self.lang_group_layer(lang_group_embedding)
                        # lang_aggregation_emb = lang_aggregation_emb.sum(dim=1) / lang_aggregation_emb.shape[1]
                        
                        valid_cnt += 1
                        
                    else:
                        pass
                        # print("xxxx")
            
            # assemble the embedding
            assemb_lang_group_embedding = torch.full((valid_cnt, max_lang_group_size, hidden_size), 0, \
                dtype=lang_emb.dtype, device=self.device)
            assemb_lang_group_mask = torch.full((valid_cnt, max_lang_group_size), 0, \
                dtype=lang_emb.dtype, device=self.device)
            
            for k, ele in enumerate(lang_group_emb_lst):
                ele_len = ele.shape[0]
                assemb_lang_group_embedding[k, :ele_len] = ele
                assemb_lang_group_mask[k, :ele_len] = 1
            
            # change the assemb_lang_group_mask to attention mask
            assemb_atten_mask = self.lang_roberta.get_extended_attention_mask(assemb_lang_group_mask, \
                assemb_lang_group_mask.size(), self.device)
            
            assemb_group_output = self.lang_group_layer(assemb_lang_group_embedding, attention_mask=assemb_atten_mask)
            
            
            assemb_group_pool_output = (assemb_group_output[0] * assemb_lang_group_mask.unsqueeze(-1)).sum(1) / assemb_lang_group_mask.sum(-1).unsqueeze(-1)
            # assemb_group_pool_output = self.lang_pooler(assemb_lang_group_mask, assemb_group_output[0])

            
            all_lang_group_emb_lst.append(assemb_group_pool_output)
            # calculate the cos distance
            # assemb_group_graph_output = torch.stack(graph_group_emb_lst)
            
            
            
            
            # def cos_distance(m1, m2):
            #     mn1 = F.normalize(m1, p=2, dim=1)
            #     mn2 = F.normalize(m2, p=2, dim=1)
            #     cos_dis = (mn1 - mn2) ** 2
            #     cos_dis = cos_dis.sum(dim=1) / 2
            #     return torch.mean(cos_dis)
            # cos_dis = cos_distance(assemb_group_pool_output, assemb_group_graph_output)
            # group_align_loss += cos_dis
        
        
        graph_emb_all = torch.stack(all_graph_group_emb_lst)
        lang_emb_all = torch.cat(all_lang_group_emb_lst, dim=0)
        assert graph_emb_all.shape[0] == lang_emb_all.shape[0]
        
        cos_sim = self.sim(graph_emb_all.unsqueeze(1), lang_emb_all.unsqueeze(0))
        ctr_sample_num = cos_sim.shape[0]
        labels = torch.arange(ctr_sample_num).long().to(self.device)
        
        group_align_loss = self.ctr_loss(cos_sim, labels) + self.ctr_loss(cos_sim.t(), labels)
        print('Group align_loss is {}'.format(group_align_loss))
        return_dict['group_lang_emb'] = lang_emb_all
        return_dict['group_graph_emb'] = graph_emb_all
        
        lang_gcn_pooler_output = self.lang_pooler(lang_gcn_attention_mask, lang_gcn_outputs)
        return_dict['lang_gcn_pooler_output'] = lang_gcn_pooler_output
        return return_dict
