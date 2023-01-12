import numpy as np
import pandas as pd
import torch
import os

from tqdm import tqdm

from run_v2 import load_config, load_data, get_optimizer_and_scheduler, load_predict_data
from utils import trim_batch

from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration

from modules.random_task2route import RandomTask2Route
from modules.routing_bart_config import RoutingBartConfig
from modules.routing_bart_v2 import MyRoutingBart
from modules.utils import initialize_weights
# identical to run_v2
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
            task_routes = task_model(task_id)

            n_layer = model.config.encoder_layers + model.config.decoder_layers
            n_expert = model.config.router_block_num
            task_routes = task_routes.reshape(n_layer, n_expert)
            # print(task_routes)

            enc_routes0 = task_routes[:model.config.encoder_layers].unsqueeze(0)
            dec_routes0 = task_routes[model.config.encoder_layers:].unsqueeze(0)
            
            # print(enc_routes0)
            # print(dec_routes0)

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
                                    use_sparse=(args.task_level_random_n == 1),
                                    )

            for input_, output in zip(batch[0], outputs):
                pred = data.decode(output)
                predictions.append(pred)

        df.loc[len(df.index)] = [data.task_name, data.metric, data.evaluate(predictions)]

    return df, np.mean(df["performance"])

def train(args, logger, model, task_model, train_data, dev_data, dev_data2, optimizer, scheduler):
    
    # a dataframe to keep track of losses
    df = pd.DataFrame(columns=["steps", "train_loss", "dev_loss", "dev_performance"])

    # initialization    
    global_batch = 0
    global_step = 0
    total_target_tokens = 0
    train_losses = []
    best_loss = 1e10
    best_avg_performance = -1.0
    stop_training = False

    # for gradient clipping
    all_parameters = list(model.parameters())
    
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

            bsz = batch[1].shape[0]

            task_routes = task_model(batch[0])
            n_layer = model.config.encoder_layers + model.config.decoder_layers
            n_expert = model.config.router_block_num
            task_routes = task_routes.reshape(bsz, n_layer, n_expert).transpose(0,1)

            enc_routes, dec_routes = task_routes[:model.config.encoder_layers], task_routes[model.config.encoder_layers:]

            loss = model(input_ids=batch[1], attention_mask=batch[2], 
                decoder_input_ids=batch[3], decoder_attention_mask=batch[4],
                block_distribution=enc_routes,
                decoder_block_distribution=dec_routes,
                is_training=True)

            # because loss is divided over the number of tokens
            total_target_tokens += torch.sum(torch.sum(batch[4])).item()

            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break

            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_batch % args.gradient_accumulation_steps == 0:

                global_step += 1

                # use a universal clipping
                # but use different learning rate for different model componenets
                torch.nn.utils.clip_grad_norm_(all_parameters, args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                task_model.zero_grad()

                if global_step % args.eval_period == 0:
                    task_model.eval()
                    model.eval()

                    output_dir_current_step = os.path.join(args.output_dir, "{}-steps".format(global_step))
                    # creating the directory in case it does not exist
                    if not os.path.exists(output_dir_current_step):
                        os.makedirs(output_dir_current_step, exist_ok=True)

                    dev_loss = validate(args, logger, model, task_model, dev_data)
                    
                    df_dev_performance, avg_performance = predict(args, logger, model, task_model, dev_data2)
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

        bsz = batch[1].shape[0]

        task_routes = task_model(batch[0])
        n_layer = model.config.encoder_layers + model.config.decoder_layers
        n_expert = model.config.router_block_num
        task_routes = task_routes.reshape(bsz, n_layer, n_expert).transpose(0,1)
        enc_routes, dec_routes = task_routes[:model.config.encoder_layers], task_routes[model.config.encoder_layers:]

        total_target_tokens += torch.sum(torch.sum(batch[4])).item()

        with torch.no_grad():

            loss = model(input_ids=batch[1], attention_mask=batch[2], 
                decoder_input_ids=batch[3], decoder_attention_mask=batch[4],
                block_distribution=enc_routes,
                decoder_block_distribution=dec_routes,
                is_training=True)

            loss = loss.detach().cpu()

        eval_losses.append(loss)

    avg_loss = np.sum(eval_losses) / total_target_tokens

    return avg_loss

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
        task_model = RandomTask2Route(args, config, args.task_level_random_n, args.taskvecs, new=True)

    else:
        # initialize from a routing transformer checkpoint (usually for evaluation)
        checkpoint_name = "best" if load_best else args.checkpoint_name

        model_path = os.path.join(args.output_dir, checkpoint_name, "model.pt")
        model.load_state_dict(torch.load(model_path))

        task_model_path = os.path.join(args.output_dir, checkpoint_name)
        task_model = RandomTask2Route(args, config, args.task_level_random_n, task_model_path, new=False)

        logger.info("Loading model from {}".format(task_model_path))

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
        task_model.to(torch.device("cuda"))

    return model, task_model
