# need to be run in the root directory of the project
# PYTHONPATH="." python daily/jun11/build_a_checkpoint2.py 

### combine trained route and a unlearned routing bart

import torch
import shutil
from modules.routing_bart_config import RoutingBartConfig
from modules.routing_bart_v2 import MyRoutingBart
from modules.utils import initialize_weights
from transformers import BartForConditionalGeneration

# load a trained model (with good routes)
config = RoutingBartConfig.from_pretrained("/home/qinyuan/LayerDrop/models/archive/may19/fisher_pca128_freeze/config.json")
config.encoder_vanilla_layers = [int(item) for item in config.encoder_vanilla_layers.split(",")] if config.encoder_vanilla_layers else []
config.decoder_vanilla_layers = [int(item) for item in config.decoder_vanilla_layers.split(",")] if config.decoder_vanilla_layers else []

model = MyRoutingBart(config)
# model.load_state_dict(torch.load("/home/qinyuan/LayerDrop/models/archive/may13/routing_randomTaskemb/best/model.pt"))
model.load_state_dict(torch.load("/home/qinyuan/LayerDrop/models/archive/may19/fisher_pca128_freeze/best/model.pt"))

# load pre-trained bart-base, and use the weights to initialize the learned model
model_old = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
initialize_weights(config, model, model_old)

filename = "/home/qinyuan/LayerDrop/models/jun19/merged_checkpoint_fisherfreeze/init/model.pt"
torch.save(model.state_dict(), filename)

shutil.copy("/home/qinyuan/LayerDrop/models/archive/may19/fisher_pca128_freeze/best/id2task.json",
        "/home/qinyuan/LayerDrop/models/jun19/merged_checkpoint_fisherfreeze/init/id2task.json")
shutil.copy("/home/qinyuan/LayerDrop/models/archive/may19/fisher_pca128_freeze/best/task_vecs.npy",
        "/home/qinyuan/LayerDrop/models/jun19/merged_checkpoint_fisherfreeze/init/task_vecs.npy")
