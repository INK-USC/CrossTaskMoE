#!/bin/sh
cd ..

DATA_DIR=data/crossfit_data

PYTHONPATH="." \
python task_emb/cli_get_text_emb.py \
--data_dir=$DATA_DIR \
--custom_tasks_splits="/home/qinyuan/CrossFit/dataloader/custom_tasks_splits/random.json"
