import os
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from fewshot_gym_singletask import NLPFewshotGymSingleTaskData

from bart import MyBart
from utils import freeze_embeds, trim_batch

from tqdm import tqdm

def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained(args.model)

    train_data = NLPFewshotGymSingleTaskData(logger, args, args.train_file, data_type="train", is_training=True)
    dev_data = NLPFewshotGymSingleTaskData(logger, args, args.dev_file, data_type="dev", is_training=False)

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    best_dev_performance = None
    test_performance = None

    best_model_state_dict = None
    
    if args.do_train:
        if args.checkpoint is not None and args.checkpoint != "None":
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            model = MyBart.from_pretrained(args.model,
                                           state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
        else:
            model = MyBart.from_pretrained(args.model)

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(model)

        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=args.total_steps)
        _, last_model_state_dict = train(args, logger, model, train_data, dev_data, optimizer, scheduler)
    
####################################calculate task embedding############################################
    args.forwardTrain = True
    last_hidden_avgs = []
    last_hidden_boss = []
    if args.forwardTrain:
        train_data_eval = NLPFewshotGymSingleTaskData(logger, args, args.train_file, data_type="train", is_training=False)
        train_data_eval.load_dataset(tokenizer)
        train_data_eval.load_dataloader()
        model.eval()
        for i, batch in enumerate(train_data_eval.dataloader):
            #print("batch_i: ",i)
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            pad_token_id = dev_data.tokenizer.pad_token_id
            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])
            logit,out = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=False,output_hidden_states=True,return_dict=True)
            last_hidden_avg = torch.mean(out['encoder_last_hidden_state'][0][1:-1],dim=0).cpu().detach().numpy()
            last_hidden_bos = out['encoder_last_hidden_state'][0][0].cpu().detach().numpy()           
            last_hidden_avgs.append(last_hidden_avg)
            last_hidden_boss.append(last_hidden_bos)
        #print(output.keys())
        #print(batch[0])
        task2vecBOS = np.mean(last_hidden_boss,axis=0)
        task2vecAVG = np.mean(last_hidden_avgs,axis=0)
        #print(task2vecBOS.shape)
        #print(output['encoder_last_hidden_state'])
        #print(task2vecAVG.shape)
        np.save(open(os.path.join(args.output_dir, "AVGtaskebd.npy"),'wb+'),task2vecAVG)
        print("successfully write in {}".format(os.path.join(args.output_dir, "AVGtaskebd.npy")))
        np.save(open(os.path.join(args.output_dir, "BOStaskebd.npy"),'wb+'),task2vecBOS)
        print("successfully write in {}".format(os.path.join(args.output_dir, "BOStaskebd.npy")))
        
        return task2vecBOS,task2vecAVG
####################################calculate task embedding############################################
        

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    best_performance = -1.0
    stop_training=False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch), disable=args.quiet):
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            
            pad_token_id = train_data.tokenizer.pad_token_id

            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])

            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True)
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            #train_losses.append(loss.detach().cpu())
            if global_step > args.total_steps-10:
                logger.info("train loss=%s , step=%s" % (loss.data,global_step))
            #print(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()
            '''
            if global_step % args.eval_period == 0:
                model.eval()
                curr_performance = inference(model if args.n_gpu==1 else model.module, dev_data)
                logger.info("Step %d Train loss %.2f %s %s on epoch=%d" % (
                        global_step,
                        np.mean(train_losses),
                        dev_data.metric,
                        curr_performance,
                        epoch))
                train_losses = []
                if best_performance < curr_performance:
                    best_model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    #model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    #torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    logger.info("Not saving model with best %s: %s -> %s on epoch=%d, global_step=%d" % \
                            (dev_data.metric, best_performance, curr_performance, epoch, global_step))
                    best_performance = curr_performance
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
    
                model.train()
            '''
            if global_step >= args.total_steps:
                stop_training = True
                break
             
        if stop_training:
            break

    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
    #torch.save(model_state_dict, os.path.join(args.output_dir, "last-model.pt"))
    return best_performance, model_state_dict

