#!/bin/sh


cd ..

DATA_DIR=data/crossfit_data_v2
SPLIT=/home/qinyuan/CrossFit/dataloader/custom_tasks_splits/random.json
OUTPUT_DIR=models/may9/test_clean_code
TASKVECS=data/taskvec/dummy

TASK_ONTOLOGY=playground/plot_utils/ontology.json

python cli_heterogenous.py \
--model facebook/bart-base \
--initialize_from_vanilla \
--do_train \
--learning_rate 1e-5 \
--router_lr 1e-3 \
--task_model_lr 1e-2 \
--total_steps=30000 \
--eval_period=750 \
--task_ontology=$TASK_ONTOLOGY \
--train_dir=$DATA_DIR \
--predict_dir=$DATA_DIR \
--taskvecs=$TASKVECS \
--custom_tasks_splits=$SPLIT \
--output_dir=$OUTPUT_DIR;
