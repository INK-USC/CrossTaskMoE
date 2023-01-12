# PYTHONPATH="." python task_emb/cli_get_text_emb.py

from my_datasets.dataloader import MyData
from utils import get_tasks_list, trim_batch

import os
import json
import numpy as np
import torch
from transformers import BartTokenizer, BartModel

from tqdm import tqdm

def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    all_tasks = get_tasks_list(args.custom_tasks_splits, "all")

    predict_data = MyData(logger, args, args.data_dir, tasks=all_tasks, data_type="train", is_training=True)
    predict_data.load_dataset(tokenizer, mode="router")
    predict_data.load_dataloader(mode="router")

    model = BartModel.from_pretrained("facebook/bart-base")
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    all_tasks, all_vecs = compute(args, logger, model, predict_data)

    with open(os.path.join(args.output_dir, "id2task.json"), "w") as fout:
        json.dump(all_tasks, fout)

    np.save(os.path.join(args.output_dir, "task_vecs.npy"), all_vecs)


def compute(args, logger, model, predict_data):

    bsz = args.predict_batch_size

    bos_token_id = predict_data.tokenizer.bos_token_id
    pad_token_id = predict_data.tokenizer.pad_token_id

    all_predictions = {}

    for idx, task in tqdm(enumerate(predict_data.data), total=len(predict_data.data)):
        start_idx, end_idx = predict_data.dataset.metadata_task[idx]

        task_name = task["task_name"]
        task_prefix = task["task_prefix"]

        input_ids_for_this_task = predict_data.dataset.input_ids[start_idx: end_idx]
        masks_for_this_task = predict_data.dataset.attention_mask[start_idx: end_idx]

        predictions = []

        for j in range(0, len(input_ids_for_this_task), bsz):
            input_ids_this_batch = input_ids_for_this_task[j: j+bsz]
            masks_for_this_batch = masks_for_this_task[j: j+bsz]

            if torch.cuda.is_available():
                input_ids_this_batch = input_ids_this_batch.to(torch.device("cuda"))
                masks_for_this_batch = masks_for_this_batch.to(torch.device("cuda"))

            input_ids_this_batch, masks_for_this_batch = trim_batch(input_ids_this_batch, pad_token_id, masks_for_this_batch)


            with torch.no_grad():
                outputs = model(input_ids=input_ids_this_batch,
                    attention_mask=masks_for_this_batch)
        
            to_save = torch.mean(outputs["last_hidden_state"], dim=1).cpu()

            if task_name in all_predictions:
                all_predictions[task_name].append(to_save)
            else:
                all_predictions[task_name] = [to_save]

    all_vecs = []
    all_tasks = sorted(list(all_predictions.keys()))
    print(all_tasks)

    for task in all_tasks:
        v = torch.mean(torch.cat(all_predictions[task], dim=0), dim=0)
        all_vecs.append(v)

    all_tasks = {i: item for i, item in enumerate(all_tasks)}
    all_vecs = torch.stack(all_vecs).numpy()
    
    return all_tasks, all_vecs