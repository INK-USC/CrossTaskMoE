import os
import json
import numpy as np

def load_taskvecs(directory):
    task_names_file = os.path.join(directory, "task_names.json")
    task_vecs_file = os.path.join(directory, "task_vecs.npy")

    with open(task_names_file) as fin:
        task_names = json.load(fin)
    task_vecs = np.load(task_vecs_file)

    return_dict = {k: v for k, v in zip(task_names, task_vecs)}

    return return_dict