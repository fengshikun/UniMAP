# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """


import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict
import tqdm
import re

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        words: list. The words of the sequence.
        positions: 2d positions  read from dict
    """
    #labels: Optional[str]
    seq: Optional[str] = None
    word_pos_dic: Optional[Dict] = None
    words: Optional[List[str]] = None
    positions: Optional[Dict] = None


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    strucpos_ids: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]] 
    #labels: Optional[int]

class Split(Enum): # cache name
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data import Dataset

    class ParsedSeqDataset(Dataset):# ParsedIUPACDataset
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            max_pos_depth: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()
            if max_pos_depth is None or 'vanilla' in task:
                cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    tokenizer.vocab_size,
                    str(max_seq_length),
                    task,
                ),
            )
            else:
                cached_features_file = os.path.join(
                    data_dir,
                    "cached_{}_{}_{}_{}_{}_{}".format(
                        mode.value,
                        tokenizer.__class__.__name__,
                        tokenizer.vocab_size,
                        str(max_seq_length),
                        str(max_pos_depth),
                        task,
                    ),
                )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    #label_list = processor.get_labels()
                    if mode == Split.dev:
                        examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    else:
                        examples = processor.get_train_examples(data_dir)
                    logger.info("Number of examples: %s", len(examples))
                    if max_pos_depth is None:
                        self.features = convert_examples_seq_to_features(
                            examples=examples,
                            max_seq_length=max_seq_length,
                            tokenizer=tokenizer,
                        )
                    else:
                        self.features = convert_examples_with_strucpos_to_features(
                            examples=examples,
                            max_seq_length=max_seq_length,
                            max_pos_depth=max_pos_depth,
                            tokenizer=tokenizer,
                        )
                    print('Checking length of features:',len(self.features)) # 
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        '''
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        '''
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")
    '''
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]
    '''

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self):
        """Gets the list of examples for this data set."""
        raise NotImplementedError()

    
class VanillaIUPACProcessor(DataProcessor):
    """Processor for the data set using VanillaIUPAC"""
    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""

        json_lines = [json.loads(l) for l in lines]
        #print(json_lines[0])
        examples = [
            InputExample(
                seq=line['iupac'], 
                #labels=list(line['atom_number'].values()), # [14,27,2,1,0,0,...] len:10
            )
            for line in json_lines  
        ]
        return examples

class VanillaSMILESProcessor(DataProcessor):
    """Processor for the data set using VanillaIUPAC"""
    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""

        json_lines = [json.loads(l) for l in lines]
        #print(json_lines[0])
        examples = [
            InputExample(
                seq=line['smiles'], 
                #labels=list(line['atom_number'].values()), # [14,27,2,1,0,0,...] len:10
            )
            for line in json_lines 
        ]
        return examples

class ParsedIUPACProcessor(DataProcessor):
    """Processor for the data set using ParsedIUPAC"""

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        '''        
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")
        '''

        json_lines = [json.loads(l) for l in lines]
        #print(json_lines[0])
        examples = [
            InputExample(
                seq=line['iupac'], 
                word_pos_dic=line['parsed_iupac'],
                words=list(line['parsed_iupac'].keys()), #['acetyloxy', 'trimethylazaniumyl', 'butanoate']
                positions=list(line['parsed_iupac'].values()), #[{'0': [3]}, {'0': [4]}, {}]
            )
            for line in json_lines 
        ]

        return examples

def convert_examples_seq_to_features(
    examples: List[InputExample],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #labels = example.labels
        seq = example.seq

        inputs = tokenizer(
                seq,
                add_special_tokens=True,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
        )
        input_ids = inputs['input_ids']
        input_mask = inputs['attention_mask']

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("input text sequence: %s", seq)
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            #logger.info("labels: %s", " ".join([str(x) for x in labels]))

        #if "token_type_ids" not in tokenizer.model_input_names:
        segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, 
                attention_mask=input_mask, 
                token_type_ids=segment_ids, 
                strucpos_ids=None,
                #labels=labels
            )
        )
    return features



def convert_examples_seq_to_features_wlen(
    examples: List[InputExample],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    features = []
    seq_length = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #labels = example.labels
        seq = example.seq

        inputs = tokenizer(
                seq,
                add_special_tokens=True,
                max_length=max_seq_length,
                # padding="max_length",
                truncation=True,
        ) # only to the truncation, no padding
        input_ids = inputs['input_ids']
        input_mask = inputs['attention_mask']
        input_len = len(input_ids)
        seq_length.append(input_len)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("input text sequence: %s", seq)
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            #logger.info("labels: %s", " ".join([str(x) for x in labels]))

        #if "token_type_ids" not in tokenizer.model_input_names:
        segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, 
                attention_mask=input_mask, 
                token_type_ids=segment_ids, 
                strucpos_ids=None,
                #labels=labels
            )
        )
    return features, seq_length


def convert_examples_with_strucpos_to_features(
    examples: List[InputExample],
    max_seq_length: int,
    max_pos_depth: int,
    tokenizer: PreTrainedTokenizer,
    cls_token="<s>",
    cls_token_segment_id=1,
    sep_token="</s>",
    sep_token_extra=False,
    pad_token=1,
    pad_token_segment_id=0,
    pad_token_pos_id=[[0,0]],
    sequence_a_segment_id=0,
    mask_padding_with_zero=True
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """


    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #labels = example.labels
        tokens = []
        strucpos_ids = []

        seq = example.seq
        word_pos_dic = example.word_pos_dic

        seq_split = re.split(r'([a-zA-Z]+)', seq)

        for part in seq_split:
            if part == '':
                continue
            else:
            #for word, position in zip(example.words, example.positions):# one word, one pos

                word_tokens = tokenizer.tokenize(part)

                # print('word_tokens',word_tokens)
                # print('position',position,type(position))

                pos_ids = []
                if part not in list(word_pos_dic.keys()) or (part in list(word_pos_dic.keys()) and len(word_pos_dic[part])==0):
                    pos_ids.extend(pad_token_pos_id*max_pos_depth) #[0,0]
                else:
                    for i, (depth, number_list) in enumerate(word_pos_dic[part].items()):
                        # print(i, k, v)
                        depth = int(depth)
                        pos_2d = [[depth,num] for num in number_list]
                        pos_ids.extend(pos_2d) #[[1,2],[3,4]]
                        
                    len_diff = len(pos_ids) - max_pos_depth
                    if len_diff>=0:
                        pos_ids = pos_ids[len_diff:] # truncate from tail
                    else:
                        pos_ids += pad_token_pos_id * (-len_diff)
                        
                pos_ids = [pos_ids]
                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens) # extend means adding the elements of parameter list
                    # Use the real label id for the tokens of the word
                    strucpos_ids.extend(pos_ids * len(word_tokens))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        # special_tokens_count = tokenizer.num_special_tokens_to_add()
        if sep_token_extra:
            special_tokens_count = 3
        else:
            special_tokens_count = 2
        # print('special_tokens_count:',special_tokens_count)
        # print('max_seq_length - special_tokens_count',max_seq_length - special_tokens_count)
        # print('len(tokens) ',len(tokens) )
        # print('tokens',tokens)
        if len(tokens) > (max_seq_length - special_tokens_count):
            #print('Cutting')
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            strucpos_ids = strucpos_ids[: (max_seq_length - special_tokens_count)]

        #print('original:',len(strucpos_ids),strucpos_ids)
        tokens += [sep_token]
        strucpos_ids += [pad_token_pos_id*max_pos_depth]
        #print('sep after:',len(strucpos_ids),strucpos_ids)
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            strucpos_ids += [pad_token_pos_id*max_pos_depth]
            #print('sep extra after:',len(strucpos_ids),strucpos_ids)
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        strucpos_ids = [pad_token_pos_id*max_pos_depth] + strucpos_ids
        #print('cls before:',len(strucpos_ids),strucpos_ids)
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids) # 
        if padding_length>0:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            strucpos_ids += [pad_token_pos_id * max_pos_depth] * padding_length
            #print('padding after:',len(strucpos_ids),strucpos_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        #print(strucpos_ids[0],len(strucpos_ids) ,max_seq_length)
        assert len(strucpos_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("strucpos_ids: %s", " ".join([str(x) for x in strucpos_ids]))
            # logger.info("labels: %s", " ".join([str(x) for x in labels]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, 
                attention_mask=input_mask, 
                token_type_ids=segment_ids, 
                strucpos_ids=strucpos_ids
            )
        )
    return features


processors = {"parsed_iupac": ParsedIUPACProcessor,'vanilla_iupac':VanillaIUPACProcessor,'vanilla_smiles':VanillaSMILESProcessor}
#MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4, "swag", 4, "arc", 4, "syn", 5}