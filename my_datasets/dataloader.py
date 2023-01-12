import os
import json
import re
import string
import random

import numpy as np
import pandas as pd

from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from .utils import load_taskvecs
from .metrics import METRICS, evaluate

class MyData(object):
    def __init__(self, logger, args, data_path, tasks, data_type, is_training):
        self.data_path = data_path

        self.data_type = data_type # train/dev/test

        self.data = []

        self.instance_count = 0

        for task in sorted(tasks):

            task_dir = os.path.join(self.data_path, task)
            files = sorted(os.listdir(task_dir))
            prefixes = []

            # list all the prefixes (e.g., "acronym_identification_32_100_")
            for filename in files:
                if not filename.endswith(".tsv"):
                    continue
                prefix = "_".join(filename.split("_")[:-1])
                if prefix not in prefixes:
                    prefixes.append(prefix)

            for prefix in prefixes:
                # load examples
                with open(os.path.join(task_dir, prefix + "_{}.tsv".format(data_type))) as fin:
                    lines = fin.readlines()
                
                examples = []
                for line in lines:
                    d = line.strip().split("\t")
                    examples.append((d[0], d[1:]))
                    # there may be multiple valid output for the same input instances, d[1:] will keep all of them

                self.data.append({
                    "task_name": task,
                    "task_prefix": prefix,
                    "examples": examples,
                })

                self.instance_count += len(examples)

        self.is_training = is_training
        self.logger = logger
        self.args = args

        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.load = True

        self.gen_early_stop = False

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        # answers is a list of lists, answers[i] is the valid outputs for the i-th input.
        # new_answers[metadata[i]: metadata[i+1]] will be the list of answers for the i-th input.
        # flatten is helpful if all answers are pre-tokenized and saved to disk.
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, mode, do_return=False):

        assert mode in ["router", "simple"]
        
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        split_identifier = self.args.custom_tasks_splits.split("/")[-1]
        if split_identifier.endswith(".json"):
            split_identifier = split_identifier[:-5]

        preprocessed_path = os.path.join(
            self.data_path,
            self.data_type + "-router-{}-{}.json".format(split_identifier, postfix)
        )
        
        if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, \
                decoder_input_ids, decoder_attention_mask, \
                metadata_task, metadata_questions = json.load(f)

        else:
            self.logger.info("Start tokenizing ... {} instances".format(len(self.data)))

            inputs = []
            outputs = []
            metadata_task, metadata_questions = [], []
            st, ed = 0, 0

            for task in self.data:
                task_name = task["task_name"]

                for dp in task["examples"]:
                    inputs.append(" [{}] {}".format(task_name, dp[0]))
                    outputs.append([" " + item for item in dp[1]])
                    # the additional whitespace is to deal with some tokenization issue

                st = ed
                ed = ed + len(task["examples"])
                metadata_task.append((st, ed))                             

            outputs, metadata_questions = self.flatten(outputs)

            self.logger.info("Printing 3 examples")
            for i in range(3):
                self.logger.info(inputs[i])
                self.logger.info(outputs[i])

            if self.args.do_lowercase:
                inputs = [input0.lower() for input0 in inputs]
                outputs = [output0.lower() for output0 in outputs]

            if self.args.append_another_bos:
                inputs = ["<s> "+input0 for input0 in inputs]
                outputs = ["<s> " +output0 for output0 in outputs]
            
            self.logger.info("Tokenizing Train Input ...")
            tokenized_input = tokenizer.batch_encode_plus(inputs,
                                                         padding="max_length",
                                                         truncation=True,
                                                         max_length=self.args.max_input_length)
            self.logger.info("Tokenizing Train Output ...")
            tokenized_output = tokenizer.batch_encode_plus(outputs,
                                                       padding="max_length",
                                                       truncation=True,
                                                       max_length=self.args.max_output_length)

            input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = tokenized_output["input_ids"], tokenized_output["attention_mask"]

            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                                decoder_input_ids, decoder_attention_mask,
                                metadata_task, metadata_questions,
                                ], f)

        all_task_names = [task["task_name"] for task in self.data]

        if mode == "router":
            self.dataset = MyTaskLevelDataset(all_task_names, input_ids, attention_mask,
                                            decoder_input_ids, decoder_attention_mask,
                                            metadata_task, metadata_questions,
                                            inner_bsz=self.args.inner_bsz,
                                            is_training=self.is_training)
        elif mode == "simple":
            self.dataset = MySimpleDataset(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, 
                                            out_metadata=metadata_questions, is_training=self.is_training)

        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, mode, do_return=False):
        assert mode in ["router", "simple"]

        if mode == "router":
            self.dataloader = MyTaskLevelDataLoader(self.args, self.dataset, self.is_training)
        elif mode == "simple":
            self.dataloader = MySimpleDataLoader(self.args, self.dataset, self.is_training)

        if do_return:
            return self.dataloader

    def evaluate(self, predictions, verbose=False):
        # return 0.0
        df = pd.DataFrame(columns=["task_name", "task_prefix", "metric", "result"])

        all_results = []

        for i, task in enumerate(self.data):
            predictions_for_this_task = predictions[i]
            predictions_for_this_task = [prediction.strip() for prediction in predictions_for_this_task]
            data_for_this_task = task["examples"]
            metric = METRICS[task["task_name"]]
            try:
                result = evaluate(predictions_for_this_task, data_for_this_task, metric)
            except Exception as e:
                print(e)
                print(task["task_name"])
                print(task["task_prefix"])
                print(predictions_for_this_task)
                print([dp[1] for dp in data_for_this_task])
                # print(data_for_this_task)
                result = 0.0

            df.loc[len(df.index)] = [task["task_name"], task["task_prefix"], metric, result]
            # all_results.append(task["task_prefix"], result)
            
        return df

class MyTaskLevelDataset(Dataset):
    def __init__(self,
        task_names,
        input_ids, attention_mask, 
        decoder_input_ids, decoder_attention_mask,
        metadata_task, metadata_questions,
        inner_bsz, 
        is_training,
    ):

        self.task_names = task_names

        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)

        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)

        self.metadata_task = metadata_task
        self.metadata_questions = metadata_questions

        self.inner_bsz = inner_bsz
        self.is_training = is_training

        # print(task_names)
        # print(len(metadata_task))
        # print(metadata_task[:10])

        assert len(self.input_ids)==len(self.attention_mask)==self.metadata_task[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.metadata_questions[-1][-1]

    def __len__(self):
        return len(self.metadata_task)

    def __getitem__(self, idx):

        task_name = self.task_names[idx]

        if self.inner_bsz <= self.metadata_task[idx][1] - self.metadata_task[idx][0]:
            in_indices = np.random.choice(range(*self.metadata_task[idx]), self.inner_bsz, replace=False)
        else:
            # if there is not enough examples in the current task, we do `sample with replacement` to fill the batch
            in_indices = np.random.choice(range(*self.metadata_task[idx]), self.inner_bsz, replace=True)

        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = [], [], [], []

        for in_index in in_indices:
            input_ids.append(self.input_ids[in_index])
            attention_mask.append(self.attention_mask[in_index])

            out_idx = np.random.choice(range(*self.metadata_questions[in_index]))

            decoder_input_ids.append(self.decoder_input_ids[out_idx])
            decoder_attention_mask.append(self.decoder_attention_mask[out_idx])

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        decoder_input_ids = torch.stack(decoder_input_ids)
        decoder_attention_mask = torch.stack(decoder_attention_mask)

        return task_name, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask

class MyTaskLevelDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size

        super(MyTaskLevelDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)
        self.collate_fn = self.dummy_collate
        self.args = args

    def dummy_collate(self, input_data):
        return input_data

    def eval_dataloader(self):
        bsz = self.args.predict_batch_size
        for idx, (start_idx, end_idx) in enumerate(self.dataset.metadata_task):
            input_ids_for_this_task = self.dataset.input_ids[start_idx: end_idx]
            masks_for_this_task = self.dataset.attention_mask[start_idx: end_idx]

            for j in range(0, len(input_ids_for_this_task), bsz):
                input_ids_this_batch = input_ids_for_this_task[j: j+bsz]
                masks_for_this_batch = masks_for_this_task[j: j+bsz]

                decoder_input_ids, decoder_attention_mask = [], []

                for in_index in range(j, min(j+bsz, end_idx-start_idx)):
                    out_idx = np.random.choice(range(*self.dataset.metadata_questions[start_idx + in_index]))
                    decoder_input_ids.append(self.dataset.decoder_input_ids[out_idx])
                    decoder_attention_mask.append(self.dataset.decoder_attention_mask[out_idx])

                decoder_input_ids = torch.stack(decoder_input_ids)
                decoder_attention_mask = torch.stack(decoder_attention_mask)
                
                assert input_ids_this_batch.shape[0] == decoder_input_ids.shape[0]

                yield self.dataset.task_names[idx], \
                    input_ids_this_batch, masks_for_this_batch, \
                    decoder_input_ids, decoder_attention_mask


class MySimpleDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        # if not self.is_training:
        #     idx = self.in_metadata[idx][0]
        #     return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]


class MySimpleDataLoader(DataLoader):
    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(MySimpleDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)


if __name__ == "__main__":
    load_taskvecs("/home/qinyuan/LayerDrop/data/taskvec/dummy")