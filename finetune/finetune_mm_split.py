import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from typing import List



import pyparsing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from absl import app, flags
import sys
sys.path.append("..")
from parsing import get_dict
#from pe_2d.utils_pe import InputExample, convert_examples_to_features
from pe_2d.utils_pe_seq import InputExample, convert_examples_seq_to_features_wlen, convert_examples_with_strucpos_to_features,convert_examples_seq_to_features
# from regex_token import IUPACTokenizer,SmilesTokenizer,
from iupac_token import IUPACTokenizer, SmilesIUPACTokenizer, SmilesTokenizer

from utils.molnet_dataloader import get_dataset_info, load_molnet_dataset


# for deep ddi task
from utils.ddi_config import DDI_CONFIG
from utils.ddi_dataset import FinetuneDDIDataset
from rdkit.Chem.PandasTools import LoadSDF
# from utils.roberta_regression import (
#     RobertaForRegression,
#     RobertaForSequenceClassification,
# )

from utils.multilingual_regression import MultilingualModal, MultilingualModalSplit, MultilingualModalUNI, MultilingualModalUNIDDI
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
InputDataClass = NewType("InputDataClass", Any)
from torch_geometric.data import Data, Batch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import TrainerCallback

import lmdb
import pickle
import yaml
from pathlib import Path

from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
    roc_auc_score,
    accuracy_score,
)
from transformers import RobertaConfig, RobertaTokenizerFast, Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from utils.dti_asset import MoleculeProteinDataset, MultilingualModalUNIDTI
from utils.mol import smiles2graph
import copy

FLAGS = flags.FLAGS

# Settings
flags.DEFINE_string(name="output_dir", default="default_dir", help="")
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_string(name="run_name", default="default_run", help="")
flags.DEFINE_integer(name="seed", default=0, help="Global random seed.")


# gcn parameters
flags.DEFINE_integer(name="gnn_number_layer", default=3, help="")
flags.DEFINE_float(name="gnn_dropout", default=0.1, help="")
flags.DEFINE_bool(name="conv_encode_edge", default=True, help="")
flags.DEFINE_integer(name="gnn_embed_dim", default=384, help="")
flags.DEFINE_string(name="gnn_aggr", default="maxminmean", help="")
flags.DEFINE_string(name="gnn_norm", default="layer", help="")
flags.DEFINE_string(name="gnn_act", default="gelu", help="")

flags.DEFINE_integer(name="graph_max_seq_size", default=128, help="")

flags.DEFINE_boolean(name="gnn_only", default=False, help="")
flags.DEFINE_boolean(name="lang_only", default=False, help="")

# for split
flags.DEFINE_boolean(name="iupac_only", default=False, help="")
flags.DEFINE_boolean(name="iupac_smiles_concat", default=False, help="")

# for uni
flags.DEFINE_boolean(name="graph_uni", default=False, help="")
flags.DEFINE_boolean(name="use_rdkit_feature", default=False, help="")
flags.DEFINE_boolean(name="use_label_weight", default=False, help="")

flags.DEFINE_boolean(name="org_reg", default=False, help="")
flags.DEFINE_boolean(name="is_debug", default=False, help="")
# use unimol data
flags.DEFINE_boolean(name="use_lmdb", default=False, help="")

flags.DEFINE_boolean(name="no_mean", default=False, help="")
flags.DEFINE_boolean(name="train_fs", default=False, help="train from scratch")
flags.DEFINE_string(name="params_config", default="", help="")
# Model params
flags.DEFINE_string(
    name="pretrained_model_name_or_path",
    default=None,
    help="Arg to HuggingFace model.from_pretrained(). Can be either a path to a local model or a model ID on HuggingFace Model Hub. If not given, trains a fresh model from scratch (non-pretrained).",
)
flags.DEFINE_boolean(
    name="freeze_base_model",
    default=False,
    help="If True, freezes the parameters of the base model during training. Only the classification/regression head parameters will be trained. (Only used when `pretrained_model_name_or_path` is given.)",
)
flags.DEFINE_boolean(
    name="is_molnet",
    default=True,
    help="If true, assumes all dataset are MolNet datasets.",
)

flags.DEFINE_boolean(
    name="use_struct",
    default=False,
    help="If true, use struct_pos ids",
)

flags.DEFINE_boolean(
    name="use_smiles",
    default=False,
    help="If true, use smiles",
)

# RobertaConfig params (only for non-pretrained models)
flags.DEFINE_integer(name="vocab_size", default=600, help="")
flags.DEFINE_integer(name="max_position_embeddings", default=515, help="")
flags.DEFINE_integer(name="num_attention_heads", default=6, help="")
flags.DEFINE_integer(name="num_hidden_layers", default=6, help="")
flags.DEFINE_integer(name="type_vocab_size", default=1, help="")


flags.DEFINE_integer(name="number_seed", default=3, help="")

flags.DEFINE_string(name="pooler_type", default="cls", help="")

# Train params
flags.DEFINE_integer(name="logging_steps", default=10, help="")
flags.DEFINE_integer(name="early_stopping_patience", default=16, help="")
flags.DEFINE_integer(name="num_train_epochs_max", default=100, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=64, help="")
flags.DEFINE_integer(name="per_device_eval_batch_size", default=64, help="")
flags.DEFINE_integer(
    name="n_trials",
    default=5,
    help="Number of different hyperparameter combinations to try. Each combination will result in a different finetuned model.",
)
flags.DEFINE_integer(
    name="n_seeds",
    default=5,
    help="Number of unique random seeds to try. This only applies to the final best model selected after hyperparameter tuning.",
)

# Dataset params
flags.DEFINE_list(
    name="datasets",
    default=None,
    help="Comma-separated list of MoleculeNet dataset names.",
)
flags.DEFINE_string(
    name="split", default="scaffold", help="DeepChem data loader split_type."
)
flags.DEFINE_list(
    name="dataset_types",
    default=None,
    help="List of dataset types (ex: classification,regression). Include 1 per dataset, not necessary for MoleculeNet datasets.",
)

# Tokenizer params
flags.DEFINE_string(
    name="tokenizer_path",
    default="seyonec/SMILES_tokenized_PubChem_shard00_160k",
    help="",
)

flags.DEFINE_string(
    name="smiles_tokenizer_path",
    default="seyonec/SMILES_tokenized_PubChem_shard00_160k",
    help=""
)

flags.DEFINE_integer(name="max_tokenizer_len", default=128, help="") # 512
flags.DEFINE_integer(name="max_concat_len", default=256, help="") # 512



flags.mark_flag_as_required("datasets")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

thecontent = pyparsing.Word(pyparsing.alphanums) #| '-' # | '+' #pyparsing.alphanums
parens = pyparsing.nestedExpr( '(', ')', content=thecontent)
def convet_iupac(iupac_line):
    line_ = iupac_line.replace('"','')\
                .replace('[','(').replace(']',')')\
                .replace(',','sep').replace('.','sep')\
                .replace('-',' ').replace('+',' ').replace(';',' ').replace('&',' ').replace('?',' ')\
                .replace('$', ' ').replace('^', ' ').replace('{', ' ').replace('}', ' ').replace('\'', ' ')
    nested_line_ = '('+line_+')'
    name_list = parens.parseString(nested_line_).asList()[0]
    name_dict = get_dict(name_list)
    return name_dict
    


def main(argv):
    print('----------------------Args check-------------------')
    print('FLAGS.use_struct ',FLAGS.use_struct,' FLAGS.use_smiles ',  FLAGS.use_smiles)
    if FLAGS.pretrained_model_name_or_path is None:
        print(
            "`WARNING: pretrained_model_name_or_path` is None - training a model from scratch."
        )
    else:
        print(
            f"Instantiating pretrained model from: {FLAGS.pretrained_model_name_or_path}"
        )

    is_molnet = FLAGS.is_molnet

    # Check that CSV dataset has the proper flags
    if not is_molnet:
        print("Assuming each dataset is a folder containing CSVs...")
        assert (
            len(FLAGS.dataset_types) > 0
        ), "Please specify dataset types for csv datasets"
        for dataset_folder in FLAGS.datasets:
            assert os.path.exists(os.path.join(dataset_folder, "train.csv"))
            assert os.path.exists(os.path.join(dataset_folder, "valid.csv"))
            assert os.path.exists(os.path.join(dataset_folder, "test.csv"))

    for i in range(len(FLAGS.datasets)):
        dataset_name_or_path = FLAGS.datasets[i]
        dataset_name = get_dataset_name(dataset_name_or_path)
        if dataset_name == 'deepddi':
            is_molnet = False
            dataset_type = 'classification'
        elif dataset_name in ['kiba', 'davis']: # dti
            dataset_type = 'regression'
            pass
        else:
            dataset_type = (
                get_dataset_info(dataset_name)["dataset_type"]
                if is_molnet
                else FLAGS.dataset_types[i]
            )

        if dataset_name != 'deepddi':
            seed_lst = [0, 1, 2]
            seed_lst = [i for i in range(FLAGS.number_seed)]
            if FLAGS.is_debug:
                seed_lst = [0]
            for seed in seed_lst:
                print("-------------Test on the seed {}-------------".format(seed))
                data_name_single = dataset_name
                dataset_name = "{}_seed{}".format(data_name_single, seed)
                run_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name, dataset_name)
                if os.path.exists(run_dir) and not FLAGS.overwrite_output_dir:
                    print(f"Run dir already exists for dataset: {dataset_name}")
                else:
                    print(f"Finetuning on {dataset_name}")
                    finetune_single_dataset(
                        dataset_name_or_path, dataset_type, run_dir, is_molnet, seed=seed
                    )
        else:
            run_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name, dataset_name)

            if os.path.exists(run_dir) and not FLAGS.overwrite_output_dir:
                print(f"Run dir already exists for dataset: {dataset_name}")
            else:
                print(f"Finetuning on {dataset_name}")
                finetune_single_dataset(
                    dataset_name_or_path, dataset_type, run_dir, is_molnet
                )


def prune_state_dict(model_dir):
    """Remove problematic keys from state dictionary"""
    if not (model_dir and os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))):
        return None

    state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
    assert os.path.exists(
        state_dict_path
    ), f"No `pytorch_model.bin` file found in {model_dir}"
    loaded_state_dict = torch.load(state_dict_path)
    # return loaded_state_dict
    state_keys = loaded_state_dict.keys()
    print('------------------prune_state_dict---------------------')
    print('state_keys: ',state_keys)
    print('-------------------------------------------------------')
    keys_to_remove = [
        k for k in state_keys if k.startswith("regression") or k.startswith("norm") or k.startswith('lm_head') or k.startswith('classifier')
    ]

    new_state_dict = OrderedDict({**loaded_state_dict})
    for k in keys_to_remove:
        del new_state_dict[k]
    return new_state_dict


def convert_to_input_examples(iupac_ids):
    input_examples = []
    for iupac_id in iupac_ids:
        line = convet_iupac(iupac_id)
        input_examples.append(InputExample(
                seq=iupac_id,
                word_pos_dic=line,
                words=list(line.keys()), #['acetyloxy', 'trimethylazaniumyl', 'butanoate']
                positions=list(line.values()), #[{'0': [3]}, {'0': [4]}, {}]
            ))
    return input_examples

def convert_to_iupac_seq_examples(iupac_ids):
    input_examples = []
    for iupac_id in iupac_ids:
        input_examples.append(InputExample(
                seq=iupac_id,
            ))
    return input_examples

def convert_to_smiles_seq_examples(smiles_ids):
    input_examples = []
    for smiles_id in smiles_ids:
        input_examples.append(InputExample(
                seq=smiles_id,
            ))
    return input_examples
        

def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    graph_lst = []
    
    left_graph_lst = []
    right_graph_lst = []
    
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, Data):
                # for deepddi
                if k == 'left_graph': 
                    left_graph_lst = [f[k] for f in features]
                elif k == 'right_graph':
                    right_graph_lst = [f[k] for f in features]
                else:
                    graph_lst = [f[k] for f in features]
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    
    if len(graph_lst):
        batch['graph'] = Batch.from_data_list(graph_lst)
    
    if len(right_graph_lst):
        batch['right_graph'] = Batch.from_data_list(right_graph_lst)
        
    if len(left_graph_lst):    
        batch['left_graph'] = Batch.from_data_list(left_graph_lst)
    
    
    return batch


def finetune_single_dataset(dataset_name, dataset_type, run_dir, is_molnet, seed=0):

    tokenizer = SmilesIUPACTokenizer.from_pretrained(
        FLAGS.tokenizer_path, max_len=FLAGS.max_tokenizer_len
    )
    iupac_tokenizer = IUPACTokenizer.from_pretrained(FLAGS.tokenizer_path, max_len=FLAGS.max_tokenizer_len)
    print('Trained tokenizer iupac vocab_size: ', tokenizer.vocab_size)
    smiles_tokenizer = SmilesTokenizer.from_pretrained(FLAGS.smiles_tokenizer_path, max_len=FLAGS.max_tokenizer_len)
    print('Trained smiles tokenizer iupac vocab_size: ', smiles_tokenizer.vocab_size)
    
    tokenizer = [iupac_tokenizer, smiles_tokenizer] # tokenizer is list

    if dataset_name == 'deepddi':
        finetune_datasets = get_ddi_datasets(dataset_name, tokenizer)
    else:
        finetune_datasets = get_finetune_datasets(dataset_name, tokenizer, is_molnet, seed=seed, use_lmdb=FLAGS.use_lmdb)

    if FLAGS.pretrained_model_name_or_path:
        config = RobertaConfig.from_pretrained(
            FLAGS.pretrained_model_name_or_path, use_auth_token=True
        )
    else:
        config = RobertaConfig(
            vocab_size=iupac_tokenizer.vocab_size,
            max_position_embeddings=FLAGS.max_position_embeddings,
            num_attention_heads=FLAGS.num_attention_heads,
            num_hidden_layers=FLAGS.num_hidden_layers,
            type_vocab_size=FLAGS.type_vocab_size,
            is_gpu=torch.cuda.is_available(),
            pooler_type=FLAGS.pooler_type,
        )
    
    smiles_config = copy.copy(config)
    smiles_config.vocab_size = tokenizer[1].vocab_size
    
    
    

    if dataset_type == "classification":
        # model_class = RobertaForSequenceClassification
        config.num_labels = finetune_datasets.num_labels
        config.num_tasks = finetune_datasets.num_tasks

    elif dataset_type == "regression":
        # model_class = RobertaForRegression
        config.num_labels = 1
        if FLAGS.org_reg:
            config.norm_mean = 0
            config.norm_std = [1]
        else:
            config.norm_mean = finetune_datasets.norm_mean
            config.norm_std = finetune_datasets.norm_std
        config.num_tasks = finetune_datasets.num_tasks

    state_dict = prune_state_dict(FLAGS.pretrained_model_name_or_path)

    def model_init():
        is_regression = (dataset_type == "regression")
        # if dataset_type == "classification":
        #     model_class = RobertaForSequenceClassification
        # elif dataset_type == "regression":
        #     model_class = RobertaForRegression
        gnn_config = {
            "gnn_number_layer": FLAGS.gnn_number_layer,
            "gnn_dropout": FLAGS.gnn_dropout,
            "conv_encode_edge": FLAGS.conv_encode_edge,
            "gnn_embed_dim": FLAGS.gnn_embed_dim,
            "gnn_aggr": FLAGS.gnn_aggr,
            "gnn_norm": FLAGS.gnn_norm,
            "gnn_act": FLAGS.gnn_act
        }
        
        if FLAGS.graph_uni:
            gnn_config['graph_max_seq_size'] = FLAGS.graph_max_seq_size
            
            if FLAGS.datasets[0] == 'deepddi':
                model_class = MultilingualModalUNIDDI
            elif FLAGS.datasets[0] in ['kiba', 'davis']:
                model_class = MultilingualModalUNIDTI
            else:
                model_class = MultilingualModalUNI
            if FLAGS.iupac_smiles_concat:
                config.vocab_size = iupac_tokenizer.vocab_size + smiles_tokenizer.vocab_size
            else:
                config.vocab_size = smiles_tokenizer.vocab_size
            model = model_class(config, gnn_config, is_regression, FLAGS.use_label_weight, FLAGS.use_rdkit_feature)
        
        elif FLAGS.iupac_smiles_concat:
            config.vocab_size = iupac_tokenizer.vocab_size + smiles_tokenizer.vocab_size
            model_class = MultilingualModal
            model = model_class(config, gnn_config, is_regression=is_regression, gnn_only=FLAGS.gnn_only, lang_only=FLAGS.lang_only)
        else:
            model_class = MultilingualModalSplit
        # model_class.from_pretrained()
            model = model_class(config, smiles_config, gnn_config, is_regression=is_regression, gnn_only=FLAGS.gnn_only, lang_only=FLAGS.lang_only, iupac_only=FLAGS.iupac_only)
        
        if not FLAGS.train_fs:
            model_class._load_state_dict_into_model(model, state_dict, FLAGS.pretrained_model_name_or_path)
        
        
        # if FLAGS.pretrained_model_name_or_path:
        #     model = model_class.from_pretrained(
        #         FLAGS.pretrained_model_name_or_path,
        #         config=config,
        #         state_dict=state_dict,
        #         use_auth_token=True,
        #     )
        #     if FLAGS.freeze_base_model:
        #         for name, param in model.base_model.named_parameters():
        #             param.requires_grad = False
        # else:
        #     model = model_class(config=config)

        return model

    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # save_strategy="no",
        output_dir=run_dir,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        per_device_eval_batch_size=FLAGS.per_device_eval_batch_size,
        logging_steps=FLAGS.logging_steps,
        load_best_model_at_end=True,
        report_to=None,
    )

    # debug
    # from utils.weighted_trainer import get_sampler_weight, WeightedTrainer

    # train_label_weight_norm = get_sampler_weight(finetune_datasets.train_dataset,
    #                                              finetune_datasets.valid_dataset)
    
    # trainer = WeightedTrainer(
    #     model_init=model_init,
    #     args=training_args,
    #     train_dataset=finetune_datasets.train_dataset,
    #     eval_dataset=finetune_datasets.valid_dataset,
    #     data_collator=default_data_collator,
    #     callbacks=[
    #         EarlyStoppingCallback(early_stopping_patience=FLAGS.early_stopping_patience)
    #     ],
    # )
    
    # trainer.set_weight(train_label_weight_norm)

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=finetune_datasets.train_dataset,
        eval_dataset=finetune_datasets.valid_dataset,
        data_collator=default_data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=FLAGS.early_stopping_patience)
        ],
    )


    # for debug
    dir_valid = os.path.join(run_dir, "results", "running_valid")
    dir_test = os.path.join(run_dir, "results", "running_test")
    os.makedirs(dir_valid, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)
    # callback for test:
    class CustomCallback(TrainerCallback):
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
            self.val_roc_lst = []
            self.test_roc_lst = []
            self.is_regress = False
        
        def set_roc_lst_clear(self):
            self.val_roc_lst = []
            self.test_roc_lst = []
        
        def get_topk_best_test_res(self, topk=1):
            
            # debug print
            print('Find topk: ')
            print('Val {}'.format(self.val_roc_lst))
            print('Test {}'.format(self.test_roc_lst))

            topk_val_idices = sorted(range(len(self.val_roc_lst)), key=lambda i: self.val_roc_lst[i], reverse=self.is_regress)[-topk:]
            test_roc_np_lst = np.array(self.test_roc_lst)
            if self.is_regress:
                return test_roc_np_lst[topk_val_idices].min()
            else:
                return test_roc_np_lst[topk_val_idices].max()
        
        def on_epoch_end(self, args, state, control, model, **kwargs):
            random_seed = 0 # dummpy value
            val_roc = eval_model(
                trainer,
                finetune_datasets.valid_dataset_unlabeled,
                dataset_name,
                dataset_type,
                dir_valid,
                random_seed,
                finetune_datasets.num_tasks
            )
            if 'roc_auc_score' not in val_roc and 'acc_score' not in val_roc:
                self.is_regress = True

            key = 'roc_auc_score'
            if 'rmse' in val_roc:
                key = 'rmse'
            elif 'acc_score' in val_roc:
                key = 'acc_score'
            

            self.val_roc_lst.append(val_roc[key])
            print("++++val_roc or rmse: {}".format(val_roc))
            test_roc = eval_model(
                trainer,
                finetune_datasets.test_dataset,
                dataset_name,
                dataset_type,
                dir_test,
                random_seed,
                finetune_datasets.num_tasks
            )
            self.test_roc_lst.append(test_roc[key])
            self.latest_val_roc = val_roc[key]
            self.latest_test_metric = test_roc[key]
            print("++++test_roc or rmse: {}".format(test_roc))
            # if control.should_evaluate:
            #     control_copy = deepcopy(control)
            #     self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            #     return control_copy
        def compute_metrics(self, eval_pred):
            return self.latest_val_roc # max for cls or min for regression

    roc_call_back = CustomCallback(trainer)
    trainer.add_callback(roc_call_back)

    def custom_hp_space_optuna(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6,1e-4, log=True),
            # "num_train_epochs": trial.suggest_int(
            #     "num_train_epochs", 20, FLAGS.num_train_epochs_max
            # ),
            # "seed": trial.suggest_int("seed", 1, 40),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [FLAGS.per_device_train_batch_size]#64,128,256
            ),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        }

    dir_valid = os.path.join(run_dir, "results", "valid")
    dir_test = os.path.join(run_dir, "results", "test")
    os.makedirs(dir_valid, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)

    metrics_valid = {}
    metrics_test = {}

    # Run with several seeds so we can see std
    # for random_seed in range(FLAGS.n_seeds):
    setattr(trainer.args, "seed", seed)
    torch.manual_seed(seed)

    if not FLAGS.params_config:
        trainer.args.num_train_epochs = 3
        direction = 'maximize'
        if dataset_type == 'regression':
            direction = 'minimize'
        best_trial = trainer.hyperparameter_search(
            backend="optuna",
            direction=direction,
            hp_space=custom_hp_space_optuna,
            n_trials=FLAGS.n_trials,
            compute_objective = roc_call_back.compute_metrics
        )
        print("Search result is: {}".format(best_trial.hyperparameters))
        # Set parameters to the best ones from the hp search
        for n, v in best_trial.hyperparameters.items():
            setattr(trainer.args, n, v)
    else:
        print("Use passed config as parameter, skip the parameter searching")
        param_conf = yaml.safe_load(Path(FLAGS.params_config).read_text())
        for n, v in param_conf.items():
            setattr(trainer.args, n, v)
        
    
    trainer.args.num_train_epochs = FLAGS.num_train_epochs_max # default 100

    if FLAGS.datasets[0] == 'deepddi':
        trainer.args.num_train_epochs = 20
    

    


    roc_call_back.set_roc_lst_clear()
    trainer.train()
    best_res = roc_call_back.get_topk_best_test_res()
    print("+++++++++Best res is: {}++++++++++++".format(best_res))
    metrics_valid[f"seed_{seed}"] = eval_model(
        trainer,
        finetune_datasets.valid_dataset_unlabeled,
        dataset_name,
        dataset_type,
        dir_valid,
        seed,
        finetune_datasets.num_tasks
    )
    metrics_test[f"seed_{seed}"] = eval_model(
        trainer,
        finetune_datasets.test_dataset,
        dataset_name,
        dataset_type,
        dir_test,
        seed,
        finetune_datasets.num_tasks
    )

    # with open(os.path.join(dir_valid, "metrics.json"), "w") as f:
    #     json.dump(metrics_valid, f)
    # with open(os.path.join(dir_test, "metrics.json"), "w") as f:
    #     json.dump(metrics_test, f)

    # Delete checkpoints from hyperparameter search since they use a lot of disk
    # for d in glob(os.path.join(run_dir, "run-*")):
    #     shutil.rmtree(d, ignore_errors=True)


def eval_model(trainer, dataset, dataset_name, dataset_type, output_dir, random_seed, num_tasks=1, train_dataset=None):
    labels = dataset.labels
    predictions = trainer.predict(dataset)
    fig = plt.figure(dpi=144)

    if dataset_type == "classification":
        unique_label = np.unique(labels)
        unique_num = len(unique_label)
        if -1 in unique_label:
            unique_num -= 1

            
        if unique_num <= 2:
            # y_pred = softmax(predictions.predictions, axis=1)[:, 1]
            if num_tasks > 1:
                
                
                valid_preds = [[] for _ in range(num_tasks)]
                valid_targets = [[] for _ in range(num_tasks)]
                for i in range(labels.shape[0]):
                    for j in range(num_tasks):
                        if dataset.task_weights[i][j] != 0:
                            valid_preds[j].append(predictions.predictions[i][j])
                            valid_targets[j].append(labels[i][j])
                
                # for every task calculate the mean metric
                results = []
                for i in range(num_tasks):
                    nan = False
                    if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                        nan = True
                    if len(valid_targets[i]) == 0:
                        nan = True
                    if nan:
                        results.append(float('nan'))
                        continue
                    y_pred = 1/(1 + np.exp(-np.array(valid_preds[i])))
                    results.append(roc_auc_score(y_true=valid_targets[i], y_score=y_pred))
                
                metrics = {
                    "roc_auc_score": np.nanmean(results)
                }
                        
               
                y_pred = 1/(1 + np.exp(-predictions.predictions))
                idx = labels.sum(axis=0) != 0
                labels = labels[:, idx]
                y_pred = y_pred[:, idx]
            else:
                # for single task
                if predictions.predictions.shape[1] > 2:
                    # multi class(deep ddi)
                    y_pred = softmax(predictions.predictions, axis=1)
                else:
                    y_pred = softmax(predictions.predictions, axis=1)[:, 1]
                    
                metrics = {
                    "roc_auc_score": roc_auc_score(y_true=labels, y_score=y_pred),
                    "average_precision_score": average_precision_score(
                        y_true=labels, y_score=y_pred
                    ),
                }
            # sns.histplot(x=y_pred, hue=labels)
        else:
            y_pred = np.argmax(predictions.predictions, axis=-1)
            metrics = {"mcc": matthews_corrcoef(labels, y_pred), 'acc_score': accuracy_score(labels, y_pred)}

    elif dataset_type == "regression":
        y_pred = predictions.predictions.flatten()
        metrics = {
            "pearsonr": pearsonr(y_pred, labels),
            "rmse": mean_squared_error(y_true=labels, y_pred=y_pred, squared=False),
        }
        # sns.regplot(x=y_pred, y=labels)
        # plt.xlabel("ChemBERTa predictions")
        # plt.ylabel("Ground truth")
    else:
        raise ValueError(dataset_type)

    # plt.title(f"{dataset_name} {dataset_type} results")
    # plt.savefig(os.path.join(output_dir, f"results_seed_{random_seed}.png"))

    return metrics



def get_ddi_datasets(dataset_name, tokenizer):
    assert dataset_name == 'deepddi'
    label_file = DDI_CONFIG.label_file
    pairs = []
    labels = []
    with open(label_file, 'r') as lr:
        lr.readline()
        for line in lr:
            info_array = line.strip().split()
            pairs.append(info_array[:2])
            labels.append(int(info_array[2]))
    
    # split to train, val, test
    data_size = len(pairs)
    pairs = np.array(pairs)
    labels = np.array(labels)
    
    num_labels = np.max(labels) # 86 for deepddi
    labels = labels - 1
    train_size = int(data_size * DDI_CONFIG.train_ratio)
    val_size = int(data_size * (1 - DDI_CONFIG.train_ratio) * 0.5)
    perm = np.random.permutation(data_size)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]
    
    num_tasks = 1
    
    # load all the smiles:
    sdf_files = glob(DDI_CONFIG.data_dir + "/*.sdf")
    smiles_dict = {}
    for sdf_file in sdf_files:
        sdf_info = LoadSDF(sdf_file, smilesName='SMILES')
        smiles = sdf_info['SMILES'].item()
        file_name = os.path.basename(sdf_file)[:-4]
        smiles_dict[file_name] = smiles
    
    # train_dataset = FinetuneDDIDataset(DDI_CONFIG.data_dir, pairs[train_idx], tokenizer[1], smiles_dict, labels[train_idx])
    
    train_dataset = FinetuneDDIDataset(DDI_CONFIG.data_dir, pairs[train_idx], tokenizer[1], smiles_dict, labels[train_idx])
    val_dataset = FinetuneDDIDataset(DDI_CONFIG.data_dir, pairs[val_idx], tokenizer[1], smiles_dict, labels[val_idx])
    
    # no label for the validation and test datasets
    # without labels
    val_dataset_unlabeled = FinetuneDDIDataset(DDI_CONFIG.data_dir, pairs[val_idx], tokenizer[1], smiles_dict, labels[val_idx], get_labels=False)
    test_dataset = FinetuneDDIDataset(DDI_CONFIG.data_dir, pairs[test_idx], tokenizer[1], smiles_dict, labels[test_idx], get_labels=False)
    
    
    # unused for ddi classification task
    norm_mean = [np.mean(np.array(train_dataset.labels), axis=0)]
    norm_std = [np.std(np.array(train_dataset.labels), axis=0)]
    
    return FinetuneDatasets(
        train_dataset,
        val_dataset,
        val_dataset_unlabeled,
        test_dataset,
        num_labels,
        num_tasks,
        norm_mean,
        norm_std,
    )
    
    
def load_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    smiles_lst = []
    target_lst = []
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        smiles_lst.append(str(data['smi']))
        target_lst.append(np.array(data['target']))
    

    return [smiles_lst, np.array(target_lst)]


def get_finetune_datasets(dataset_name, tokenizer, is_molnet, use_lmdb=False, seed=0):
    # use_lmdb = True
    # dti_task = True
    # dataset_name = 'kiba'
    if dataset_name in ['kiba', 'davis']:
        root = '/share/project/sharefs-test_data/GraphMVP/datasets/dti_datasets'
        smiles_tokenizer = tokenizer[1]
        train_dataset = MoleculeProteinDataset(root, dataset_name, smiles_tokenizer, 'train', include_labels=True)
        valid_dataset = MoleculeProteinDataset(root, dataset_name, smiles_tokenizer, 'test', include_labels=True)
        valid_dataset_unlabeled = MoleculeProteinDataset(root, dataset_name, smiles_tokenizer, 'test', include_labels=False)
        num_tasks = 1
        if FLAGS.no_mean:
            norm_mean = [0]
            norm_std = [1]
        else:
            norm_mean = [float(np.mean(np.array(train_dataset.labels), axis=0))]
            norm_std = [float(np.std(np.array(train_dataset.labels), axis=0))]
        num_labels = 2
        return FinetuneDatasets(
            train_dataset,
            valid_dataset,
            valid_dataset_unlabeled,
            valid_dataset_unlabeled, # test save with valid
            num_labels,
            num_tasks,
            norm_mean,
            norm_std,
        )
    if use_lmdb:
        lmdb_dir = '/sharefs/sharefs-test_data/Uni-Mol/molecular_property_prediction'
        train_df = load_lmdb(os.path.join(lmdb_dir, dataset_name, "train.lmdb"))
        valid_df = load_lmdb(os.path.join(lmdb_dir, dataset_name, "valid.lmdb"))
        test_df = load_lmdb(os.path.join(lmdb_dir,dataset_name, "test.lmdb"))
        tasks_wanted = None
    elif is_molnet:
        tasks_wanted, (train_df, valid_df, test_df), _ = load_molnet_dataset(
            dataset_name, split=FLAGS.split, df_format="chemprop", seed=seed,
        )
        # assert len(tasks_wanted) == 1
    else:
        train_df = pd.read_csv(os.path.join(dataset_name, "train.csv"))
        valid_df = pd.read_csv(os.path.join(dataset_name, "valid.csv"))
        test_df = pd.read_csv(os.path.join(dataset_name, "test.csv"))

    train_dataset = FinetuneDataset(train_df, tokenizer, use_struct_pos=FLAGS.use_struct, tasks_wanted=tasks_wanted, iupac_only=FLAGS.iupac_only, lang_only=FLAGS.lang_only, gnn_only=FLAGS.gnn_only, iupac_smiles_concat=FLAGS.iupac_smiles_concat, graph_uni=FLAGS.graph_uni, use_rdkit_feature=FLAGS.use_rdkit_feature, use_lmdb=use_lmdb)
    valid_dataset = FinetuneDataset(valid_df, tokenizer, use_struct_pos=FLAGS.use_struct, tasks_wanted=tasks_wanted, iupac_only=FLAGS.iupac_only, lang_only=FLAGS.lang_only, gnn_only=FLAGS.gnn_only, iupac_smiles_concat=FLAGS.iupac_smiles_concat, graph_uni=FLAGS.graph_uni, use_rdkit_feature=FLAGS.use_rdkit_feature, use_lmdb=use_lmdb)
    # valid_dataset = FinetuneDataset(test_df, tokenizer, use_struct_pos=FLAGS.use_struct, tasks_wanted=tasks_wanted, use_smiles=FLAGS.use_smiles)
    valid_dataset_unlabeled = FinetuneDataset(valid_df, tokenizer, include_labels=False, use_struct_pos=FLAGS.use_struct, tasks_wanted=tasks_wanted, iupac_only=FLAGS.iupac_only, lang_only=FLAGS.lang_only, gnn_only=FLAGS.gnn_only, iupac_smiles_concat=FLAGS.iupac_smiles_concat, graph_uni=FLAGS.graph_uni, use_rdkit_feature=FLAGS.use_rdkit_feature, use_lmdb=use_lmdb)
    test_dataset = FinetuneDataset(test_df, tokenizer, include_labels=False, use_struct_pos=FLAGS.use_struct, tasks_wanted=tasks_wanted, iupac_only=FLAGS.iupac_only, lang_only=FLAGS.lang_only, gnn_only=FLAGS.gnn_only, iupac_smiles_concat=FLAGS.iupac_smiles_concat, graph_uni=FLAGS.graph_uni, use_rdkit_feature=FLAGS.use_rdkit_feature, use_lmdb=use_lmdb)

    num_labels = len(np.unique(train_dataset.labels))
    if -1 in np.unique(train_dataset.labels):
        num_labels -= 1
    all_labels = np.concatenate([train_dataset.labels, valid_dataset.labels, test_dataset.labels])
    # norm_mean = [np.mean(np.array(train_dataset.labels), axis=0)]
    # norm_std = [np.std(np.array(train_dataset.labels), axis=0)]
    norm_mean = [np.mean(all_labels)]
    norm_std = [np.std(all_labels)]
    # for debug:
    # norm_mean = [0]
    # norm_std = [1]


    num_tasks = 1
    if len(train_dataset.labels.shape) == 2:
        num_tasks = train_dataset.labels.shape[1]

    return FinetuneDatasets(
        train_dataset,
        valid_dataset,
        valid_dataset_unlabeled,
        test_dataset,
        num_labels,
        num_tasks,
        norm_mean,
        norm_std,
    )


def get_dataset_name(dataset_name_or_path):
    return os.path.splitext(os.path.basename(dataset_name_or_path))[0]


@dataclass
class FinetuneDatasets:
    train_dataset: str
    valid_dataset: torch.utils.data.Dataset
    valid_dataset_unlabeled: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    num_labels: int
    num_tasks: int
    norm_mean: List[float]
    norm_std: List[float]




from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from rdkit import Chem
from typing import Callable, Union
Molecule = Union[str, Chem.Mol]

def rdkit_2d_features_normalized_generator(mol: Molecule) -> np.ndarray:
    """
    Generates RDKit 2D normalized features for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D normalized features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]
    return features


import multiprocessing
from tqdm import tqdm


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, include_labels=True, use_struct_pos=True, tasks_wanted=None, iupac_only=False, lang_only=False, gnn_only=False, iupac_smiles_concat=False, graph_uni=False, use_rdkit_feature=False, use_lmdb=False):
        #df = df[(df['iupac_ids'] != 'Request error') &  (df['iupac_ids'] != '')] # assume df has iupac field
        # df = df[(df['iupac_ids'] != 'Request error') &  (df['iupac_ids'] != '') & (pd.isna(df['iupac_ids']) == False)] # assume df has iupac field
        self.iupac_only = iupac_only
        self.lang_only = lang_only
        self.gnn_only = gnn_only
        
        self.use_struct_pos = use_struct_pos
        
        # concat smiles and iupac as one sequence
        self.iupac_smiles_concat = iupac_smiles_concat
        self.graph_uni = graph_uni
        # self.iupac_ids = False
        # if 'iupac_ids' in df.keys():
            
            # filter
        if use_lmdb:
            smiles_ids = df[0]
            self.labels = df[1]
        else: # dataframe
            if 'smile_ids' not in df.columns:
                smiles_ids = df['smiles'].tolist()
            else:
                smiles_ids = df["smile_ids"].tolist()
        
        
        self.fp_features = None
        if use_rdkit_feature:
            self.fp_features = []
            pool = multiprocessing.Pool(24)
            smiles_lst = smiles_ids
            total = len(smiles_lst)
            
            for res in tqdm(pool.imap(rdkit_2d_features_normalized_generator, smiles_lst, chunksize=10), total=total):
                replace_token = 0
                fp_feature = np.where(np.isnan(res), replace_token, res)
                self.fp_features.append(np.float32(fp_feature))    
        
        
        # for smile in df["smile_ids"].tolist():
        #     fp_feature = rdkit_2d_features_normalized_generator(smile)
        #     replace_token = 0
        #     fp_feature = np.where(np.isnan(fp_feature), replace_token, fp_feature)
        #     self.fp_features.append(np.float32(fp_feature))
        
        
        if self.lang_only:
            self.smiles_lst = smiles_ids # for uni
            if self.iupac_smiles_concat:
                input_examples = convert_to_iupac_seq_examples(df["iupac_ids"].tolist())
                self.iupac_features, self.iupac_length = convert_examples_seq_to_features_wlen(input_examples, max_seq_length=128,tokenizer=tokenizer[0]) 
                input_examples = convert_to_smiles_seq_examples(smiles_ids)
                self.smiles_features, self.smiles_length = convert_examples_seq_to_features_wlen(input_examples, max_seq_length=128,tokenizer=tokenizer[1])

                self.iupac_start_emb = tokenizer[1].vocab_size # smiles token size of the begining

                self.labels = df.iloc[:, 2].values

            elif self.iupac_only:
                if use_struct_pos:
                    input_examples = convert_to_input_examples(df["iupac_ids"].tolist())
                    self.features = convert_examples_with_strucpos_to_features(input_examples, max_seq_length=128,
                                max_pos_depth=16, tokenizer=tokenizer[0])
                else:
                    input_examples = convert_to_iupac_seq_examples(df["iupac_ids"].tolist())
                    self.features = convert_examples_seq_to_features(input_examples, max_seq_length=128,tokenizer=tokenizer[0]) 
                    # self.encodings = tokenizer(df["smiles"].tolist(), truncation=True, padding=True)
                self.iupac_ids = True
                self.labels = df.iloc[:, 2].values
            else:
                input_examples = convert_to_smiles_seq_examples(smiles_ids)
                self.encodings = convert_examples_seq_to_features(input_examples, max_seq_length=128,tokenizer=tokenizer[1])
                if not use_lmdb:
                    self.labels = df.iloc[:, 1].values
        elif self.gnn_only:
            # save smiles_ids
            self.smiles_lst = smiles_ids
        else:
            raise NotImplementedError
    
        
        if tasks_wanted is not None:
            if len(tasks_wanted) == 1:
                self.labels = df[tasks_wanted[0]].values
            else:
                labels = []
                for task in tasks_wanted:
                    labels.append(df[task].values.reshape(-1, 1))
                self.labels = np.concatenate(labels, axis=1)
            
        if use_lmdb and self.labels.shape[1] > 1: # multitask
            # task_weight
            self.task_weights = (self.labels != -1).astype(np.float32)
        else:
            task_weight_cols = list(df.columns[1:-2]) # X ,..., smiles_ids, iupac_ids
            self.task_weights = None

            if 'w' in task_weight_cols or 'w1' in task_weight_cols:
                # import pdb; pdb.set_trace()
                task_weights = []
                for task in tasks_wanted:
                    task_idx = task_weight_cols.index(task)
                    task_weight_col_name = 'w' + str(task_idx + 1)
                    if len(task_weight_cols) == 2:
                        task_weight_col_name = task_weight_cols[1]
                    task_weights.append(df[task_weight_col_name].tolist())
                
                task_weights = np.array(task_weights, dtype=np.float32)
                
                self.task_weights = task_weights.T
        self.include_labels = include_labels


    def _concat_lang(self, smiles_inputs, iupac_inputs, smile_len, iupac_len, concat_max_length=256):
        return_dict = {}
        # import pdb;pdb.set_trace()
        return_dict['input_ids'] = torch.full((1, concat_max_length), 1)[0] # padding: 1
        return_dict['attention_mask'] = torch.full((1, concat_max_length), 0)[0] # attention mask, default: 0
        

        return_dict['input_ids'][:smile_len] = torch.tensor(smiles_inputs.input_ids[:smile_len])
        
        iupac_input_ids = torch.tensor(iupac_inputs.input_ids[1:iupac_len]) # earse the iupac cls token
        iupac_input_ids[:-1] += self.iupac_start_emb # except the sep token
        return_dict['input_ids'][smile_len: smile_len + iupac_len - 1] = iupac_input_ids
        
        return_dict['attention_mask'][:smile_len] = torch.tensor(smiles_inputs.attention_mask[:smile_len])
        return_dict['attention_mask'][smile_len: smile_len + iupac_len - 1] = torch.tensor(iupac_inputs.attention_mask[1:iupac_len]) # erase the iupac cls token    
        return return_dict


    def __getitem__(self, idx):
        # get smiles and transfer to graph
        if self.gnn_only:
            item = {}
            smiles = self.smiles_lst[idx]
            graph, _ = smiles2graph(smiles)
            item['graph'] = graph
        elif self.lang_only:
            if self.iupac_smiles_concat:
                item = {}
                # concat smiles and iupac features
                item = self._concat_lang(self.smiles_features[idx], self.iupac_features[idx], \
                    self.smiles_length[idx], self.iupac_length[idx], FLAGS.max_concat_len)
            elif self.iupac_only:
                if self.use_struct_pos:
                    item = {}
                    item['input_ids']=self.features[idx].input_ids
                    item['attention_mask']=self.features[idx].attention_mask
                    item['strucpos_ids']=self.features[idx].strucpos_ids
                else:
                    #item = {key: torch.tensor(val[idx]) for key, val in self.features.items()}
                    item = {}
                    item['input_ids']=self.features[idx].input_ids
                    item['attention_mask']=self.features[idx].attention_mask
            else:
                item = {}
                item['input_ids']=self.encodings[idx].input_ids
                item['attention_mask']=self.encodings[idx].attention_mask
            
            if self.graph_uni:
                smiles = self.smiles_lst[idx]
                graph, _ = smiles2graph(smiles)
                item['graph'] = graph # add graph 
        else:
            raise NotImplementedError
        
        if self.include_labels and self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
            if self.task_weights is not None:
                item['weight'] = torch.tensor(self.task_weights[idx], dtype=torch.float)
        
        if self.fp_features is not None:
            item['fp_feature'] = self.fp_features[idx] 
        
        return item

    def __len__(self):
        if self.gnn_only:
            return len(self.smiles_lst)
        if self.iupac_smiles_concat:
            return len(self.iupac_features)

        if self.iupac_only:
            return len(self.features)#["input_ids"])
        return len(self.encodings)#["input_ids"])


if __name__ == "__main__":
    app.run(main)