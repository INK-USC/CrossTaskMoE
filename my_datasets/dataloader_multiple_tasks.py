"""
Basically a list of tuples of SingleTask
[(train, dev), (train, dev), (train, dev)]
"""

import os

from .dataloader_single_task import MySingleTaskData

class MyMultipleTasksData(object):
    def __init__(self, logger, args, data_path, tasks):

        self.all_tasks = []
        self.prefix2id = {}
        cnt = 0

        for task in sorted(tasks):
            task_path = os.path.join(data_path, task)

            files = sorted(os.listdir(task_path))
            prefixes = []
            
            for filename in files:
                if not filename.endswith(".tsv"):
                    continue
                prefix = "_".join(filename.split("_")[:-1])
                if prefix not in prefixes:
                    prefixes.append(prefix)

            for prefix in prefixes:
                train_file = os.path.join(task_path, prefix + "_train.tsv")
                dev_file = os.path.join(task_path, prefix + "_dev.tsv")

                self.all_tasks.append((
                    MySingleTaskData(logger, args, train_file, "train", is_training=True, verbose=False),
                    MySingleTaskData(logger, args, dev_file, "dev", is_training=False, verbose=False)
                ))
                self.prefix2id[prefix] = cnt
                cnt += 1

    def __getitem__(self, idx):
        return self.all_tasks[idx]

    def load_dataset(self, tokenizer):
        for train_data, dev_data in self.all_tasks:
            train_data.load_dataset(tokenizer)
            dev_data.load_dataset(tokenizer)

    def load_dataloader(self):
        for train_data, dev_data in self.all_tasks:
            train_data.load_dataloader()
            dev_data.load_dataloader()





