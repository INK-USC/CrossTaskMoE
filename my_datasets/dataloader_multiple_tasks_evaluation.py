"""
Basically a list of SingleTask
"""

import os

from .dataloader_single_task import MySingleTaskData

class MyMultipleTasksEvaluationData(object):
    def __init__(self, logger, args, data_path, tasks, split):

        assert split in ["train", "dev", "test"]

        self.tasks = tasks
        self.all_tasks = []
        cnt = 0

        for task in sorted(tasks):
            task_path = os.path.join(data_path, task)

            files = sorted(os.listdir(task_path))
            valid_filenames = list(filter(lambda x: x.endswith("_{}.tsv".format(split)), files))
            assert len(valid_filenames) == 1
            filename = os.path.join(task_path, valid_filenames[0])

            self.all_tasks.append(MySingleTaskData(logger, args, filename, data_type=split, is_training=False, verbose=False, task_name=task))

    def __getitem__(self, idx):
        return self.all_tasks[idx]

    def __len__(self):
        return len(self.tasks)

    def load_dataset(self, tokenizer):
        for data in self.all_tasks:
            data.load_dataset(tokenizer)

    def load_dataloader(self):
        for data in self.all_tasks:
            data.load_dataloader()





