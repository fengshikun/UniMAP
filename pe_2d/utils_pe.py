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
import numpy as np

import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        words: list. The words of the sequence.
        positions: 2d positions  read from dict
    """

    words: List[str]
    positions: Dict


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    strucpos_ids: List[List[int]]
    token_type_ids: Optional[List[List[int]]] = None

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data import Dataset

    class ParsedIUPACDataset(Dataset):
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

            cached_features_file = os.path.join(
                data_dir,
                "cached_refined_{}_{}_{}_{}_{}_{}".format(
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
                    logger.info("Training examples: %s", len(examples))
                    self.features = convert_examples_to_features(
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
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    '''
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    '''
    

class IUPACProcessor(DataProcessor):
    """Processor for the SWAG data set."""

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

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        '''        
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")
        '''

        json_lines = [json.loads(l) for l in lines]
        examples = [
            InputExample(
                words=list(line.keys()), #['acetyloxy', 'trimethylazaniumyl', 'butanoate']
                positions=list(line.values()), #[{'0': [3]}, {'0': [4]}, {}]
            )
            for line in json_lines  # we skip the line with the column names [1:]
        ]

        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    max_seq_length: int,
    max_pos_depth: int,
    tokenizer: PreTrainedTokenizer,
    cls_token="<s>",
    cls_token_segment_id=1,
    sep_token="</s>",
    sep_token_extra=True,
    pad_token=1,
    pad_token_segment_id=0,
    pad_token_pos_id=[[0,0]],#[[-1,-1]],
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

        tokens = []
        strucpos_ids = []
        
        for word, position in zip(example.words, example.positions):# one word, one pos

            word_tokens = tokenizer.tokenize(word)

            # print('word_tokens',word_tokens)
            # print('position',position,type(position))

            pos_ids = []
            if len(position) == 0:
                pos_ids.extend(pad_token_pos_id*max_pos_depth) #[-1,-1]
            else:
                for i, (depth, number_list) in enumerate(position.items()):
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
        special_tokens_count = 3
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

        strucpos_array = np.array(strucpos_ids)
        #print((strucpos_array<0).sum())

        #if ex_index < 5:
        if (strucpos_array<0).sum() >0:
            logger.info("*** Example ***")
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("strucpos_ids: %s", " ".join([str(x) for x in strucpos_ids]))

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


'''
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features
'''


processors = {"IUPAC": IUPACProcessor}
#MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4, "swag", 4, "arc", 4, "syn", 5}