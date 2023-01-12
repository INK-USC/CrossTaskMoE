import torch
import torch.nn as nn

import os
import json
import numpy as np

class RandomTask2Route(nn.Module):
    def __init__(self, args, config, top_n, init_directory, new=False):
        super(RandomTask2Route, self).__init__()

        self.id2task, taskvecs = load_taskvecs(init_directory)
        self.task2id = {v: int(k) for k, v in self.id2task.items()}

        self.n_tasks = taskvecs.shape[0]
        self.dim = (config.encoder_layers + config.decoder_layers) * config.router_block_num

        self.embed = nn.Embedding(self.n_tasks, self.dim)

        if new:
            random_routes = random_sample_route(
                n_tasks = self.n_tasks, 
                n_layer = config.encoder_layers + config.decoder_layers,
                n_expert = config.router_block_num,
                top_n = top_n
            )
            self.embed.load_state_dict({"weight": random_routes})
        else:
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

def random_sample_route(n_tasks, n_layer, n_expert, top_n):
    assert top_n <= n_expert - 1
    
    m = np.zeros((n_tasks, n_layer, n_expert))
    selection = np.arange(n_expert)

    for i in range(n_tasks):
        for j in range(n_layer):
            selected = np.random.choice(n_expert, top_n, replace=False)
            m[i,j,selected] = 1.0 / top_n

    m = m.reshape(n_tasks, n_layer * n_expert)
    return torch.from_numpy(m)

if __name__ == "__main__":
    t = random_sample_route(120, 12, 3, 1)
    print(t)

    t = random_sample_route(120, 12, 3, 2)
    print(t)    