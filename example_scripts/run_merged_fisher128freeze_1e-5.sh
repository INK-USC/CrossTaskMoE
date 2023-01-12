#!/bin/sh

cd ../..

ROUTER=linear_layer_specific

DATA_DIR=data/crossfit_data_v2
PREDICT_DATA_DIR=data/crossfit_data_test
SPLIT=/home/qinyuan/CrossFit/dataloader/custom_tasks_splits/random.json
OUTPUT_DIR=models/jun19/merged_checkpoint_fisherfreeze_1e-5
TASKVECS=models/jun19/merged_checkpoint_fisherfreeze_1e-5/init
TASK_ONTOLOGY=playground/plot_utils/ontology.json

python cli_heterogenous.py \
--model facebook/bart-base \
--checkpoint_name init \
--do_train \
--do_predict \
--learning_rate 1e-5 \
--router_lr 0.0 \
--task_model_lr 0.0 \
--initial_tau 300.0 \
--minimum_tau 1.0 \
--anneal_rate 0.0001 \
--router_mode softmax \
--task_ontology=$TASK_ONTOLOGY \
--total_steps=60000 \
--no_eval_until=15000 \
--eval_period=3000 \
--train_dir=$DATA_DIR \
--predict_dir=$PREDICT_DATA_DIR \
--custom_tasks_splits=$SPLIT \
--taskvecs=$TASKVECS \
--taskvecs_dim=128 \
--output_dir=$OUTPUT_DIR;
