import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch

from collections import OrderedDict

from modules.routing_bart_config import RoutingBartConfig
from modules.routing_bart_v2 import MyRoutingBart
from modules.task2vec import Task2Vec

def get_tasks_list(filename, split_name):
    with open(filename, "r") as fin:
        split_dict = json.load(fin)
    return split_dict[split_name]

def get_task2id(taskvecs_dir):
    with open(os.path.join(taskvecs_dir, "id2task.json"), "r") as fin:
        id2task = json.load(fin)

    task2id = {v: int(k) for k,v in id2task.items()}
    return task2id

def get_ontology(filename):
    with open(filename, "r") as fin:
        ontology = json.load(fin)

    reverse_ontology = dict()
    for first_cat in ontology.keys():
        for second_cat in ontology[first_cat].keys():
            for taskname in ontology[first_cat][second_cat]:
                reverse_ontology[taskname]=(second_cat,first_cat)

    return OrderedDict(ontology), OrderedDict(reverse_ontology)

def prune_ontology(tasks, ontology):
    for k, v in ontology.items():
        for k1, v1 in v.items():
            v[k1] = sorted(filter(lambda x: x in tasks, v1))

    count_dir = [(0, "None")]
    count_subdir = [(0, "None")]
    sorted_tasks = []

    for k, v in ontology.items():
        for k1, v1 in v.items():
            count_subdir.append((count_subdir[-1][0] + len(v1), k1))
            sorted_tasks += v1
        count_dir.append((count_subdir[-1][0], k))

    return ontology, sorted_tasks, count_dir, count_subdir

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def plot(weights, tasks, filename, reverse_ontology,horizontation=True):
    # weights: n_task x n_layer x n_router
    weights2D = torch.transpose(weights.reshape([len(tasks),-1]),0,1).numpy() # 36 * n_task
    index = [['expert1_'+str(i+1),'expert2_'+str(i+1),'expert3_'+str(i+1)] for i in range(12)]
    index = [i for j in index for i in j]

    df = pd.DataFrame(weights2D, columns=tasks, index=index)

    task_cat2=sorted(list(set([reverse_ontology[task][0] for task in tasks]))) # second category names
    task_cat1=sorted(list(set([reverse_ontology[task][1] for task in tasks]))) # first category names

    types2=sns.hls_palette(len(list(task_cat2)),h=0.1, l=0.6, s=0.65)
    types1=sns.color_palette("Blues_r", 1)+sns.color_palette("Reds_r", 1)+sns.color_palette("Greens_r", 1)+sns.color_palette("Purples_r", 1)

    pattern1 = dict(zip(task_cat1, types1))
    pattern2 = dict(zip(task_cat2, types2))

    lut2 = dict()
    lut1 = dict()
    for task in tasks:
        lut2[task]=pattern2[reverse_ontology[task][0]]
        lut1[task]=pattern1[reverse_ontology[task][1]]

    col_colors2 = df.columns.map(lut2)
    col_colors1 = df.columns.map(lut1)

    cmap="Blues"
    if horizontation:
        figsize=(weights2D.shape[1]//4,weights2D.shape[0]//2)

        g = sns.clustermap(df,figsize=figsize,cmap="Blues",
                           row_cluster=False,col_cluster=False,
                           xticklabels=True, yticklabels=True,
                           col_colors=[col_colors2,col_colors1],linewidths = 0.05,
                           vmin=0, vmax=1
                           )
    else:
        figsize=(weights2D.shape[1]//8,weights2D.shape[0]//2)
        g = sns.clustermap(df.T,figsize=figsize,cmap="Blues",
                           row_cluster=False,col_cluster=False,
                           xticklabels=True, yticklabels=True,
                           row_colors=[col_colors2,col_colors1],linewidths = 0.05,
                           vmin=0, vmax=1, cbar_pos=(0.175,0.89,0.6,0.005),cbar_kws=dict(orientation='horizontal')
                           )

    handles1 = [Patch(facecolor=pattern1[name]) for name in pattern1]
    handles2 = [Patch(facecolor=pattern2[name]) for name in pattern2]
    if horizontation:
        l1 = plt.legend(handles1, pattern1, title='First_Categary',
                bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='best')
        l2 = plt.legend(handles2, pattern2, title='Second_Categary', ncol=6,
                bbox_to_anchor=(0.8, 1), bbox_transform=plt.gcf().transFigure, loc='best')
        plt.gca().add_artist(l1)
    else:
        l1 = plt.legend(handles1, pattern1, title='First_Categary', ncol=4,
                        bbox_to_anchor=(0.25, 0.86), bbox_transform=plt.gcf().transFigure, loc='center left')
        plt.legend(handles2, pattern2, title='Second_Categary', ncol=6,
                        bbox_to_anchor=(0.75, 0.85), bbox_transform=plt.gcf().transFigure, loc='upper right',fontsize='x-small')
        plt.gca().add_artist(l1)

    ax = g.ax_heatmap
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=7)
    ax.hlines([3,6,9,12,15,18,21,24,27,30,33], *ax.get_xlim(), colors="k")

    plt.savefig(filename, dpi=150)

def get_gumbel_temperature(args, steps):
    return np.maximum(
        args.initial_tau * np.exp(-args.anneal_rate * steps),
        args.minimum_tau
    )

def load_saved_checkpoint(path, logger):
    # load task embeddings
    task_model = Task2Vec(os.path.join(path, "best"))

    # load main transformer
    config = RoutingBartConfig.from_pretrained(os.path.join(path, "config.json"))
    model = MyRoutingBart(config)

    logger.info("loading model from {}".format(path))
    model.load_state_dict(torch.load(os.path.join(path, "best", "model.pt")))

    return task_model, model
