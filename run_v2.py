import os
import numpy as np
import pandas as pd
import math
import torch
import json

from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup

from my_datasets.dataloader_heterogenous import MyHeterogenousData
from my_datasets.dataloader_multiple_tasks_evaluation import MyMultipleTasksEvaluationData

from modules.routing_bart_config import RoutingBartConfig
from modules.routing_bart_v2 import MyRoutingBart
from modules.task2vec import Task2Vec
from modules.utils import initialize_weights

from utils import trim_batch, get_tasks_list, get_task2id, freeze_params, plot, get_gumbel_temperature, prune_ontology, get_ontology

from tqdm import tqdm

def run(args, logger):

    config = load_config(args, logger)
    model, task_model = load_model(args, config, logger)

    if args.do_train:
        train_data, dev_data, dev_data2 = load_data(args, logger)
        optimizer, scheduler = get_optimizer_and_scheduler(args, logger, model, task_model, train_data)

        train(args, logger, model, task_model, train_data, dev_data, dev_data2, optimizer, scheduler)

        # if we will do prediction, we will load the best checkpoint in the training process
        if args.do_predict:
            model, task_model = load_model(args, config, logger, load_best=True)

    if args.do_predict:
        dev_data, test_data = load_predict_data(args, logger)

        model.eval()
        task_model.eval()
        model.set_gumbel_temperature(args.minimum_tau)
        model.set_router_mode(args.router_mode)

        df, avg_performance = predict(args, logger, model, task_model, dev_data)
        logger.info("[Eval] Dev average performance: {}".format(avg_performance))
        df.to_csv(os.path.join(args.output_dir, args.checkpoint_name, "eval-dev-performance.csv"))

        df, avg_performance = predict(args, logger, model, task_model, test_data)
        logger.info("[Eval] Test average performance: {}".format(avg_performance))
        df.to_csv(os.path.join(args.output_dir, args.checkpoint_name, "eval-test-performance.csv"))

def predict(args, logger, model, task_model, predict_data):

    df = pd.DataFrame(columns=["task_prefix", "metric", "performance"])

    for data in tqdm(predict_data):

        with torch.no_grad():
            task_id = task_model.taskname2id(data.task_name)
            if torch.cuda.is_available():
                task_id = task_id.to(torch.device("cuda"))
            task_emb = task_model(task_id)
            enc_routes0, dec_routes0 = model.get_routes(task_emb, separate=True)

        predictions = []
        bos_token_id = data.tokenizer.bos_token_id
        for i, batch in enumerate(data.dataloader):
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            pad_token_id = data.tokenizer.pad_token_id
            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            bsz = batch[0].shape[0]
            enc_routes = enc_routes0.expand(bsz, -1, -1).transpose(0,1)
            dec_routes = dec_routes0.expand(bsz, -1, -1).transpose(0,1)

            outputs = model.generate(input_ids=batch[0],
                                    attention_mask=batch[1],
                                    block_distribution=enc_routes,
                                    decoder_block_distribution=dec_routes,
                                    num_beams=data.args.num_beams,
                                    max_length=data.args.max_output_length,
                                    decoder_start_token_id=model.config.bos_token_id,
                                    early_stopping=data.gen_early_stop,
                                    use_cache=True,
                                    use_sparse=("gumbel" in args.router_mode),
                                    )

            for input_, output in zip(batch[0], outputs):
                pred = data.decode(output)
                predictions.append(pred)

        df.loc[len(df.index)] = [data.task_name, data.metric, data.evaluate(predictions)]

    return df, np.mean(df["performance"])

def train(args, logger, model, task_model, train_data, dev_data, dev_data2, optimizer, scheduler):
    
    # a dataframe to keep track of losses
    df = pd.DataFrame(columns=["steps", "train_loss", "dev_loss", "dev_performance"])

    os.makedirs(os.path.join(args.output_dir, "dev_performance_logs"), exist_ok=True)

    # initialization    
    global_batch = 0
    global_step = 0
    total_target_tokens = 0
    train_losses = []
    best_loss = 1e10
    best_avg_performance = -1.0
    stop_training = False

    # for gradient clipping
    all_parameters = list(model.parameters()) + list(task_model.parameters())
    
    # router mode will be from [softmax, gumbel_softmax, gumbel_softmax_st]
    model.set_router_mode(args.router_mode)
    # if "gumbel" in args.router_mode:
    model.set_gumbel_temperature(args.initial_tau)

    logger.info("Starting training!")

    model.train()
    task_model.train()

    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch)):

            global_batch += 1

            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            
            pad_token_id = train_data.tokenizer.pad_token_id

            batch[1], batch[2] = trim_batch(batch[1], pad_token_id, batch[2])
            batch[3], batch[4] = trim_batch(batch[3], pad_token_id, batch[4])

            task_embeds = task_model(batch[0])

            loss = model(input_ids=batch[1], attention_mask=batch[2], 
                decoder_input_ids=batch[3], decoder_attention_mask=batch[4],
                task_embed=task_embeds,
                is_training=True)

            # because loss is divided over the number of tokens
            total_target_tokens += torch.sum(torch.sum(batch[4])).item()

            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break

            train_losses.append(loss.detach().cpu())

            if args.expert_regularization > 0:
                reg_loss = model.get_weight_regularization_loss()
                loss = loss + args.expert_regularization * reg_loss

            loss.backward()

            if global_batch % args.gradient_accumulation_steps == 0:

                global_step += 1

                # use a universal clipping
                # but use different learning rate for different model componenets
                torch.nn.utils.clip_grad_norm_(all_parameters, args.max_grad_norm)

                if args.alternate_every > 0:
                    if (global_step // args.alternate_every) % 2 == 1:
                        for parameter in model.router.parameters():
                            parameter.grad.fill_(0.0)
                        for parameter in task_model.parameters():
                            parameter.grad.fill_(0.0)
                    else:
                        for parameter in model.model.parameters():
                            parameter.grad.fill_(0.0)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                task_model.zero_grad()

                # after each gradient step we update the gumbel temperature
                # if "gumbel" in args.router_mode:
                tau = get_gumbel_temperature(args, global_step)
                model.set_gumbel_temperature(tau)

                if global_step % args.eval_period == 0 and global_step >= args.no_eval_until:
                    task_model.eval()
                    model.eval()

                    output_dir_current_step = os.path.join(args.output_dir, "{}-steps".format(global_step))
                    # creating the directory in case it does not exist
                    if not os.path.exists(output_dir_current_step):
                        os.makedirs(output_dir_current_step, exist_ok=True)

                    # plot the routing paths to a .npy file and a .png file (for visualization)
                    analyze(args, logger, model, task_model, output_dir_current_step)

                    df_dev_performance, avg_performance = predict(args, logger, model, task_model, dev_data2)
                    dev_loss = validate(args, logger, model, task_model, dev_data)
                    df_dev_performance.to_csv(os.path.join(output_dir_current_step, "dev_results.csv"))
                    
                    if avg_performance > best_avg_performance:
                        logger.info("Saving model with best dev avg performance: %s -> %s at epoch=%d, global_step=%d" % \
                            (best_avg_performance, avg_performance, epoch, global_step))
                        best_avg_performance = avg_performance

                        save_path = os.path.join(args.output_dir, "best")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path, exist_ok=True)

                        # save main model
                        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                        torch.save(model_state_dict, os.path.join(save_path, "model.pt"))
                        # save task vecs
                        task_model.save_to_disk(save_path)

                    avg_train_loss = np.sum(train_losses)/total_target_tokens
                    df.loc[len(df.index)] = [global_step, avg_train_loss, dev_loss, avg_performance]
                    df.to_csv(os.path.join(args.output_dir, "losses.csv"))

                    logger.info("Step {}: train loss: {}, dev loss: {}".format(global_step, avg_train_loss, dev_loss))
                    
                    task_model.train()
                    model.train()

                    total_target_tokens = 0
                    train_losses = []

                if global_step % args.save_period == 0:

                    # create a subdirectory
                    directory_name = "{}-steps".format(global_step)
                    save_path = os.path.join(args.output_dir, directory_name)

                    if not os.path.exists(save_path):
                        os.makedirs(save_path, exist_ok=True)

                    logger.info("Checkpoint at step {} saved to {}".format(global_step, save_path))

                    # save main model
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(save_path, "model.pt"))
                    # save task vecs
                    task_model.save_to_disk(save_path)

            if global_step >= args.total_steps:
                stop_training = True
                break

        if stop_training:
            break

    # save the last model
    save_path = os.path.join(args.output_dir, "last")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
    torch.save(model_state_dict, os.path.join(save_path, "model.pt"))
    task_model.save_to_disk(save_path)

    return model, task_model

def validate(args, logger, model, task_model, eval_data):

    eval_losses = []
    total_target_tokens = 0

    for batch in tqdm(eval_data.dataloader, desc="Eval"):

        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        
        pad_token_id = eval_data.tokenizer.pad_token_id

        batch[1], batch[2] = trim_batch(batch[1], pad_token_id, batch[2])
        batch[3], batch[4] = trim_batch(batch[3], pad_token_id, batch[4])

        total_target_tokens += torch.sum(torch.sum(batch[4])).item()

        with torch.no_grad():

            task_embeds = task_model(batch[0])

            loss = model(input_ids=batch[1], attention_mask=batch[2], 
                decoder_input_ids=batch[3], decoder_attention_mask=batch[4],
                task_embed=task_embeds,
                is_training=True)
            loss = loss.detach().cpu()

        eval_losses.append(loss)

    avg_loss = np.sum(eval_losses) / total_target_tokens

    return avg_loss
    # return np.random.random()

def analyze(args, logger, model, task_model, output_dir):

    # load the list of tasks used (same as in train)
    all_tasks = get_tasks_list(args.custom_tasks_splits, "train")

    # ontology: category name -> sub-category name -> task name
    # reverse_ontology: task name -> (category, sub-category)
    ontology, reverse_ontology = get_ontology(args.task_ontology)

    # pruned_ontology: category name -> sub-category name -> task name (with tasks that are in the tasks list)
    pruned_ontology, sorted_tasks, _, _ = prune_ontology(all_tasks, ontology)

    # get routes 
    model.set_router_mode("softmax")
    ret_t1 = analyze_one_model(sorted_tasks, model, task_model, temperature=1.0)
    ret = analyze_one_model(sorted_tasks, model, task_model, temperature=None)

    model.set_router_mode(args.router_mode)

    # save routes and the heatmap
    np.save(os.path.join(output_dir, "route.npy"), ret.numpy())
    plot(ret, sorted_tasks, os.path.join(output_dir, "route.png"), reverse_ontology)
    np.save(os.path.join(output_dir, "route_t1.npy"), ret_t1.numpy())
    plot(ret_t1, sorted_tasks, os.path.join(output_dir, "route_t1.png"), reverse_ontology)


def analyze_one_model(tasks, model, task_model, bsz=16, temperature=None):

    all_outputs = []

    for start_idx in range(0, len(tasks), bsz):
        end_idx = min(start_idx + bsz, len(tasks))

        task_ids = [task_model.taskname2id(item) for item in tasks[start_idx: end_idx]]
        task_ids = torch.tensor(task_ids)
    
        if torch.cuda.is_available():
            task_ids = task_ids.to(torch.device("cuda"))

        with torch.no_grad():
            task_embeds = task_model(task_ids)
            routes = model.get_routes(task_embeds, temperature=temperature)

        all_outputs.append(routes)       

    return torch.transpose(torch.cat(all_outputs, dim=1),0,1).cpu().detach() # n_task x n_layer x n_router


def load_config(args, logger):
    config_path = os.path.join(args.output_dir, "config.json")
    # if there is a saved config file in output_dir, use it
    if os.path.exists(config_path):
        config = RoutingBartConfig.from_pretrained(config_path)
        logger.info("Loading config from {}".format(config_path))
    # otherwise take the default model config and override several fields with the input args
    else:
        config = RoutingBartConfig.from_pretrained(args.model)
        config.router_nn_type = args.router_nn_type
        config.router_input_dim = args.taskvecs_dim
        config.encoder_vanilla_layers = args.encoder_vanilla_layers
        config.decoder_vanilla_layers = args.decoder_vanilla_layers
        config.exploration_epsilon = args.exploration_epsilon
        config.save_pretrained(args.output_dir)

    config.encoder_vanilla_layers = [int(item) for item in config.encoder_vanilla_layers.split(",")] if config.encoder_vanilla_layers else []
    config.decoder_vanilla_layers = [int(item) for item in config.decoder_vanilla_layers.split(",")] if config.decoder_vanilla_layers else []
    
    return config

def load_model(args, config, logger, load_best=False):
    model = MyRoutingBart(config)
    
    if not load_best and args.initialize_from_vanilla:
        # initialize from a vanilla bart checkpoint on the disk
        if args.vanilla_checkpoint is not None:
            # load the weights from a given model BartForConditionalGeneration model.
            model_old = BartForConditionalGeneration.from_pretrained(args.model, 
                                                                    state_dict=torch.load(args.vanilla_checkpoint))
            # copy and paste into multiple experts
            initialize_weights(config, model, model_old, eps=args.init_eps)
            logger.info("Initializing {} model from {}".format(args.model, args.vanilla_checkpoint))
        # initialize from a vanilla bart checkpoint from the hub
        else:
            # initialize from pre-trained bart model
            model_old = BartForConditionalGeneration.from_pretrained(args.model)
            initialize_weights(config, model, model_old, eps=args.init_eps)
            logger.info("Initializing {} model from the hub".format(args.model))

        # either way the taskvecs should be initialized from the args.taskvecs
        task_model = Task2Vec(args.taskvecs)

    else:
        # initialize from a routing transformer checkpoint (usually for evaluation)
        checkpoint_name = "best" if load_best else args.checkpoint_name

        model_path = os.path.join(args.output_dir, checkpoint_name, "model.pt")
        model.load_state_dict(torch.load(model_path))

        task_model_path = os.path.join(args.output_dir, checkpoint_name)
        task_model = Task2Vec(task_model_path)

        logger.info("Loading model from {}".format(task_model_path))

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
        task_model.to(torch.device("cuda"))

    return model, task_model

def load_data(args, logger):

    task2id = get_task2id(args.taskvecs)
    train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
    logger.info("Loading data of the following tasks: {}".format(train_tasks))

    tokenizer = BartTokenizer.from_pretrained(args.model)

    train_data = MyHeterogenousData(logger, args, args.train_dir, tasks=train_tasks, task_split="train", data_type="train", is_training=True)
    train_data.load_dataset(tokenizer, task2id)
    train_data.load_dataloader()

    dev_data = MyHeterogenousData(logger, args, args.train_dir, tasks=train_tasks, task_split="train", data_type="dev", is_training=False)
    dev_data.load_dataset(tokenizer, task2id)
    dev_data.load_dataloader()

    dev_data2 = MyMultipleTasksEvaluationData(logger, args, args.predict_dir, tasks=train_tasks, split="dev")
    dev_data2.load_dataset(tokenizer)
    dev_data2.load_dataloader()

    return train_data, dev_data, dev_data2

def load_predict_data(args, logger):
    train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
    logger.info("Loading data of the following tasks: {}".format(train_tasks))

    tokenizer = BartTokenizer.from_pretrained(args.model)

    dev_data = MyMultipleTasksEvaluationData(logger, args, args.predict_dir, tasks=train_tasks, split="dev")
    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    test_data = MyMultipleTasksEvaluationData(logger, args, args.predict_dir, tasks=train_tasks, split="test")
    test_data.load_dataset(tokenizer)
    test_data.load_dataloader()

    return dev_data, test_data

def get_optimizer_and_scheduler(args, logger, model, task_model, train_data):
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        # tranformer params + decay
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not "router" in n], 'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        # transformer params + no decay
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not "router" in n], 'weight_decay': 0.0, 'lr': args.learning_rate},
        # router params + decay
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "router" in n], 'weight_decay': args.weight_decay, 'lr': args.router_lr},
        # router params + no decay
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "router" in n], 'weight_decay': 0.0, 'lr': args.router_lr},
    ]

    # task emebedding params
    if not args.freeze_task_model:
        optimizer_grouped_parameters.append(
            {'params': task_model.parameters(), 'weight_decay': args.weight_decay, 'lr': args.task_model_lr}
        )
    else:
        freeze_params(task_model)

    # construct optimizer
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)

    # construct scheduler
    steps_per_epoch = math.ceil(len(train_data.dataloader) / args.gradient_accumulation_steps)
    warmup_steps = int(args.total_steps * args.warmup_ratio)
    scheduler =  get_linear_schedule_with_warmup(optimizer,
                                    num_warmup_steps=warmup_steps,
                                    num_training_steps=args.total_steps)
                                
    # print some useful information
    logger.info("#Batches={}, #Steps per epoch={}, #Total steps={}, #Warmup steps={}".format(
        len(train_data.dataloader), steps_per_epoch, args.total_steps, warmup_steps
    ))
    logger.info("Batch size={}, Gradient Accumulation={}".format(args.train_batch_size, args.gradient_accumulation_steps))
    
    return optimizer, scheduler
