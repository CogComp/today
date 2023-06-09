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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader

import copy

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.trainer import SequentialDistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler

from overrides_new import DataCollatorForLanguageModeling, DoNothingDataCollator, DoNothingDataCollatorForGeneration, \
    T5ForConditionalGenerationTREO
from torch.utils.data.dataset import Dataset
import numpy as np

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    loss_alpha: Optional[int] = field(
        default=None, metadata={"help": "weight the loss of regular temporal dataset"}
    )
    loss_beta: Optional[int] = field(
        default=None, metadata={"help": "weight the loss of TODAY"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_data_file_1: Optional[str] = field(
        default=None, metadata={"help": "The input TODAY training data file (a text file)."}
    )
    train_data_file_2: Optional[str] = field(
        default=None, metadata={"help": "The input regular temporal training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path_1: str, file_path_2: str):
        # assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from prob dataset file at %s", file_path_1)
        logger.info("Creating features from origin dataset file at %s", file_path_2)

        # load today
        with open(file_path_1, encoding="utf-8") as f:
            lines_1 = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # load regular temporal dataset
        with open(file_path_2, encoding="utf-8") as f:
            lines_2 = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        originals = []
        labels_1 = []
        for l in lines_1:
            originals.append(l.split("\t")[0])
            labels_1.append("answer: positive")

        self.inputs_1 = tokenizer.batch_encode_plus(originals, pad_to_max_length=True)

        n = 4
        for key, value in self.inputs_1.items():
            self.inputs_1[key] = [value[i:i + n] for i in range(0, len(value), n)]

        self.labels_1 = tokenizer.batch_encode_plus(labels_1, pad_to_max_length=True)

        n = 4
        for key, value in self.labels_1.items():
            self.labels_1[key] = [value[i:i + n] for i in range(0, len(value), n)]

        originals = []
        labels_2 = []
        for l in lines_2:
            if len(l.split("\t")) < 2:
                continue
            originals.append(l.split("\t")[0])
            labels_2.append(l.split("\t")[1])

        self.inputs_2 = tokenizer.batch_encode_plus(originals, pad_to_max_length=True)

        n = 2
        for key, value in self.inputs_2.items():
            self.inputs_2[key] = [value[i:i + n] for i in range(0, len(value), n)]

        self.labels_2 = tokenizer.batch_encode_plus(labels_2, pad_to_max_length=True)

        n = 2
        for key, value in self.labels_2.items():
            self.labels_2[key] = [value[i:i + n] for i in range(0, len(value), n)]

        self.inputs = dict()
        self.labels = dict()

        if len(self.inputs_1[key][0][0]) >= len(self.inputs_2[key][0][0]):
            extra_padding_num = len(self.inputs_1[key][0][0]) - len(self.inputs_2[key][0][0])
            extra_padding_tag = 'inputs_1'
        else:
            extra_padding_num = len(self.inputs_2[key][0][0]) - len(self.inputs_1[key][0][0])
            extra_padding_tag = 'inputs_2'

        # combine inputs_1 and inputs_2 as a data sample
        print('number of today data: ',len(self.inputs_1[key]),', number of regular data: ',len(self.inputs_2[key]))
        if extra_padding_tag == 'inputs_2':
            for key, value in self.inputs_1.items():
                for j in range(int(len(self.inputs_1[key]) / len(self.inputs_2[key])) + 1):
                    if j == int(len(self.inputs_1[key]) / len(self.inputs_2[key])):
                        # print(j)
                        self.inputs[key] += [
                            [a + extra_padding_num * [0] for a in value[i + j * (len(self.inputs_2[key]))]] +
                            self.inputs_2[key][i] for i in
                            range(0, len(self.inputs_1[key]) - len(self.inputs[key]))]
                    else:
                        self.inputs[key] += [
                            [a + extra_padding_num * [0] for a in value[i + j * (len(self.inputs_2[key]))]] +
                            self.inputs_2[key][i] for i in
                            range(0, len(self.inputs_2[key]))]

        elif extra_padding_tag == 'inputs_1':
            for key, value in self.inputs_1.items():
                for j in range(int(len(self.inputs_1[key]) / len(self.inputs_2[key])) + 1):
                    if j == int(len(self.inputs_1[key]) / len(self.inputs_2[key])):
                        self.inputs[key] += [
                            value[i + j * (len(self.inputs_2[key]))] + [a + extra_padding_num * [0] for a in
                                                                        self.inputs_2[key][i]] for i in
                            range(0, len(self.inputs_1[key]) - len(self.inputs[key]))]
                    else:
                        self.inputs[key] += [
                            value[i + j * (len(self.inputs_2[key]))] + [a + extra_padding_num * [0] for a in
                                                                        self.inputs_2[key][i]] for i in
                            range(0, len(self.inputs_2[key]))]

        for key, value in self.labels_1.items():
            for j in range(int(len(self.labels_1[key]) / len(self.labels_2[key])) + 1):
                if j == int(len(self.labels_1[key]) / len(self.labels_2[key])):
                    self.labels[key] += [value[i + j * (len(self.labels_2[key]))] + self.labels_2[key][i] for i
                                         in
                                         range(0, len(self.labels_1[key]) - len(self.labels[key]))]
                else:
                    self.labels[key] += [value[i + j * (len(self.labels_2[key]))] + self.labels_2[key][i] for i
                                         in
                                         range(0, len(self.labels_2[key]))]

        self.len = len(self.inputs["input_ids"])

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, i):
        source_ids = self.inputs["input_ids"][i]
        target_ids = self.labels["input_ids"][i]
        src_mask = self.inputs["attention_mask"][i]
        target_mask = self.labels["attention_mask"][i]
        return {"input_ids": source_ids, "attention_mask": src_mask, "lm_labels": target_ids,
                "decoder_attention_mask": target_mask}


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    # file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        ret = LineByLineTextDataset(tokenizer=tokenizer, file_path_1=args.train_data_file_1,
                                    file_path_2=args.train_data_file_2)
        print("DATA SIZE: ")
        print(len(ret))
        return ret
    else:
        return None


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    # print(TrainingArguments.logging_steps)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # print(training_args.logging_steps)
    training_args.logging_steps = 1

    # print(training_args.logging_steps)
    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path != "new":
        model = T5ForConditionalGenerationTREO.from_pretrained(
            model_args.model_name_or_path, model_args.loss_alpha, model_args.loss_beta
        )
        print('config.loss_alpha: ', model_args.loss_alpha, ', config.loss_beta: ', model_args.loss_beta)
    else:
        config = AutoConfig.from_pretrained("t5-small")
        model = T5ForConditionalGenerationTREO(config=config)

    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = DoNothingDataCollator()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        # trainer.train(model_path=model_path)
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        model.eval()
        data_collator = DoNothingDataCollatorForGeneration()
        sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=training_args.eval_batch_size,
            collate_fn=data_collator.collate_batch,
            shuffle=False,
        )
        output_eval_file = os.path.join(training_args.output_dir, 'eval_results_lm.txt')
        writer = open(output_eval_file, "w")
        for inputs in tqdm(data_loader, "Prediction"):
            for k, v in inputs.items():
                inputs[k] = v.cuda()

            with torch.no_grad():
                outputs, possitive_score, negative_score = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=12
                )

                possitive_score = possitive_score.cpu().tolist()
                negative_score = negative_score.cpu().tolist()

                dec = [tokenizer.decode(ids) for ids in outputs]

                for i in range(0, len(dec)):
                    writer.write(dec[i] + "\t" + str(possitive_score[i]) + "\t" + str(negative_score[i]) + "\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
