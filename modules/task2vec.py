import torch
import torch.nn as nn

import os
import json
import numpy as np

class Task2Vec(nn.Module):
    def __init__(self, init_directory):
        super(Task2Vec, self).__init__()

        self.id2task, taskvecs = load_taskvecs(init_directory)
        self.task2id = {v: int(k) for k, v in self.id2task.items()}

        self.n_tasks, self.dim = taskvecs.shape

        self.embed = nn.Embedding(self.n_tasks, self.dim)
        self.embed.load_state_dict({"weight": taskvecs})

    def forward(self, idx):
        return self.embed(idx)

    def taskname2id(self, name):
        return torch.tensor(self.task2id[name])

    def save_to_disk(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        task_names_file = os.path.join(directory, "id2task.json")
        task_vecs_file = os.path.join(directory, "task_vecs.npy")

        with open(task_names_file, "w") as fout:
            json.dump(self.id2task, fout)
        
        task_vecs = self.embed.weight.detach().cpu().numpy()
        np.save(task_vecs_file, task_vecs)

def load_taskvecs(directory):
    task_names_file = os.path.join(directory, "id2task.json")
    task_vecs_file = os.path.join(directory, "task_vecs.npy")

    with open(task_names_file) as fin:
        task2id = json.load(fin)
    task_vecs = torch.from_numpy(np.load(task_vecs_file))

    return task2id, task_vecs