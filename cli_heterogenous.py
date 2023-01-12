# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random
import numpy as np
import torch

from run_v2 import run
from run_avg import run as run_avg
from run_random import run as run_random
from run_task_level_random import run as run_task_level_random

def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--train_dir", default="data")
    parser.add_argument("--predict_dir", default="data")
    parser.add_argument("--model", default="facebook/bart-base", required=False)
    
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')

    ## Meta Learn parameters
    parser.add_argument('--custom_tasks_splits', type=str, default=None)
    parser.add_argument('--taskvecs', type=str, default=None, required=False)
    parser.add_argument('--taskvecs_dim', type=int, default=768)
    parser.add_argument('--task_ontology', type=str, default=None, required=False)

    ## Gumbel parameters
    parser.add_argument('--router_mode', type=str, default="gumbel_softmax_st")
    parser.add_argument('--router_nn_type', type=str, default="linear_layer_specific")
    parser.add_argument('--initial_tau', type=float, default=2.0)
    parser.add_argument('--minimum_tau', type=float, default=0.1)
    parser.add_argument('--anneal_rate', type=float, default=0.0003)

    ## Co-efficients
    parser.add_argument('--router_lr', type=float, default=1e-3)
    parser.add_argument('--task_model_lr', type=float, default=1e-3)
    parser.add_argument('--init_eps', type=float, default=1e-8)

    ## Model parameters
    parser.add_argument("--vanilla_checkpoint", type=str, default=None)
    parser.add_argument("--initialize_from_vanilla", action='store_true', default=False)
    parser.add_argument("--checkpoint_name", type=str, default="best")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument("--freeze_task_model", action='store_true', default=False)
    parser.add_argument("--encoder_vanilla_layers", type=str, default="")
    parser.add_argument("--decoder_vanilla_layers", type=str, default="")
    parser.add_argument("--expert_regularization", type=float, default=0.0)
    parser.add_argument("--exploration_epsilon", type=float, default=0.0)

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos", action='store_true', default=False)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warmup steps = total_steps * warmup_ratio.")
    parser.add_argument("--total_steps", default=100000, type=int,
                        help="Linear warmup over warmup_steps.")

    # Other parameters
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=150,
                        help="Evaluate & save model")
    parser.add_argument('--no_eval_until', type=int, default=0,
                        help="For the first few steps don't run eval to save time")
    parser.add_argument('--save_period', type=int, default=10000000000,
                        help="Evaluate & save model. By default we only save the best and the last checkpoints")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--avg_routing", action='store_true', default=False)
    parser.add_argument("--random_routing", action='store_true', default=False)
    parser.add_argument("--task_level_random_routing", action='store_true', default=False)
    parser.add_argument("--task_level_random_n", type=int, default=1)
    parser.add_argument("--alternate_every",type=int, default=0)

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    ##### Start writing logs

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_dir:
            raise ValueError("If `do_train` is True, then `train_dir` must be specified.")

    if args.do_predict:
        if not args.predict_dir:
            raise ValueError("If `do_predict` is True, then `predict_dir` must be specified.")

    logger.info("Using {} gpus".format(args.n_gpu))

    if args.avg_routing:
        run_avg(args, logger)
    elif args.random_routing:
        run_random(args, logger)
    elif args.task_level_random_routing:
        run_task_level_random(args, logger)
    else:
        run(args, logger)

if __name__=='__main__':
    main()
