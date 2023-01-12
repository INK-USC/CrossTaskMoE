import os
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup

from fewshot_gym_singletask import NLPFewshotGymSingleTaskData

from bart_with_adapter import BartWithAdapterConfig,MyBartWithAdapter
from utils import freeze_embeds, trim_batch

from tqdm import tqdm

def initialize_weights(config, model_new, model_old, eps=1e-8):
    """model_old is bart model, of which parameters are reused, while adapters are individually initialized."""

    # embeddings
    model_new.model.shared.load_state_dict(model_old.model.shared.state_dict())
    model_new.model.encoder.embed_positions.load_state_dict(model_old.model.encoder.embed_positions.state_dict())
    model_new.model.decoder.embed_positions.load_state_dict(model_old.model.decoder.embed_positions.state_dict())

    # layer norms
    model_new.model.encoder.layernorm_embedding.load_state_dict(model_old.model.encoder.layernorm_embedding.state_dict())
    model_new.model.decoder.layernorm_embedding.load_state_dict(model_old.model.decoder.layernorm_embedding.state_dict())

    # encoders
    for layer in range(config.encoder_layers):
        old_encoder_layer = model_old.model.encoder.layers[layer]

        # copy weights from the old model
        new_encoder_layer = model_new.model.encoder.layers[layer]
        new_encoder_layer.load_state_dict(old_encoder_layer.state_dict(),strict=False)

    # decoders
    for layer in range(config.decoder_layers):
        old_decoder_layer = model_old.model.decoder.layers[layer]

        # copy weights from the old model
        new_decoder_layer = model_new.model.decoder.layers[layer]
        new_decoder_layer.load_state_dict(old_decoder_layer.state_dict(),strict=False)

def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained(args.model)

    train_data = NLPFewshotGymSingleTaskData(logger, args, args.train_file, data_type="train", is_training=True)
    dev_data = NLPFewshotGymSingleTaskData(logger, args, args.dev_file, data_type="dev", is_training=False)

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()
    
    config = BartWithAdapterConfig.from_pretrained("facebook/bart-base")
    model = MyBartWithAdapter(config)

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
            model_old = BartForConditionalGeneration.from_pretrained("facebook/bart-base", 
                                                                    state_dict=torch.load(args.checkpoint))
            initialize_weights(config, model, model_old)
            logger.info("Initializing model from {}".format(args.checkpoint))
        else:
			# initialize from pre-trained bart model
            model_old = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
            initialize_weights(config, model, model_old)
        
        excludes = ['adapter']
        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            excludes = ['adapter']
            freeze_embeds(model,excludes)

        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(ic in n for ic in ['adapter'])], 'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(ic in n for ic in ['adapter'])], 'weight_decay': 0.0, 'lr': args.learning_rate} ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=args.total_steps)
        _, last_model_state_dict = train(args, logger, model, train_data, dev_data, optimizer, scheduler,config)

#######################################################calculate task embedding with fisher information############################
    model.eval()
    args.forwardTrain = True
    all_grads = []   
    includes = ['adapter_','weight']
    if args.forwardTrain:
        train_data_eval = NLPFewshotGymSingleTaskData(logger, args, args.train_file, data_type="train", is_training=False)
        train_data_eval.load_dataset(tokenizer)
        train_data_eval.load_dataloader()
        
        for n, p in model.named_parameters():
                p.requires_grad = True
                p.grad_accumu = torch.zeros_like(p.data)
                p.grad_accumu_count = 0

        for i, batch in enumerate(train_data_eval.dataloader):
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            pad_token_id = train_data_eval.tokenizer.pad_token_id
            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])
            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True,output_hidden_states=True,return_dict=True)
            model.zero_grad()
            loss.backward()
            for n, p in model.named_parameters():
                if all(ic in n for ic in includes):
                    if p.grad is None:
                        raise ValueError(f'gradient of {n} is none')
                    p.grad_accumu += p.grad.data ** 2
                    p.grad_accumu_count += 1
        for n, p in model.named_parameters():
            if all(ic in n for ic in includes):
                print(n)
                grad = (p.grad_accumu / p.grad_accumu_count)
                all_grads.append(grad.reshape(-1))
        all_grads = torch.stack(all_grads).cpu().detach().numpy()
        np.save(open(os.path.join(args.output_dir, "adapter"+str(config.adapter_dim)+"_fisher.npy"),'wb+'),all_grads)
        print("successfully write in {}".format(os.path.join(args.output_dir, "adapter"+str(config.adapter_dim)+"_fisher.npy")))
        return all_grads
#######################################################calculate task embedding with fisher information############################
       

def train(args, logger, model, train_data, dev_data, optimizer, scheduler, config):
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
            if global_step > args.total_steps-10:
                logger.info("adapter dim=%s, train loss=%s , step=%s" % (config.adapter_dim,loss.data,global_step))
            #print(loss.detach().cpu())
            #train_losses.append(loss.detach().cpu())
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

