import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from .routing_bart_config import RoutingBartConfig

class RouterWrapper(nn.Module):
    def __init__(self, config: RoutingBartConfig):
        super().__init__()
        self.config = config
        
        assert config.router_nn_type in ["linear", "rnn", "lstm", "gru", "transformer", "rnn_layer_specific", "lstm_layer_specific", "gru_layer_specific", "transformer_layer_specific", "linear_layer_specific"]

        if config.router_nn_type == "linear":
            self.model = LinearRouter(config)
        elif config.router_nn_type in ["rnn", "lstm", "gru"]:
            self.model = RecurrentRouter(config)
        elif config.router_nn_type == "transformer":
            self.model = TransformerRouter(config)
        elif config.router_nn_type in ["rnn_layer_specific", "lstm_layer_specific", "gru_layer_specific"]:
            self.model = RecurrentLayerSpecificRouter(config)
        elif config.router_nn_type == "transformer_layer_specific":
            self.model = TransformerLayerSpecificRouter(config)
        elif config.router_nn_type == "linear_layer_specific":
            self.model = LinearLayerSpecificRouter(config)
    
    def forward(self, task_embed):
        return self.model(task_embed)

class LinearLayerSpecificRouterUnit(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        # let's use two layer perceptron for now, this can be any architecture
        self.linear1 = nn.Linear(config.router_input_dim, config.router_hidden_dim)
        self.linear2 = nn.Linear(config.router_hidden_dim, config.router_block_num)
        self.activation_fn = F.gelu 

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)

        return x
        
class LinearLayerSpecificRouter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder_layers = config.encoder_layers
        self.decoder_layers = config.decoder_layers
        self.n_blocks = config.router_block_num
        self.all_layers = config.encoder_layers + config.decoder_layers

        self.router_units = nn.ModuleList(
            LinearLayerSpecificRouterUnit(config) for i in range(self.all_layers)
        )

    def forward(self, task_embed):
        # task_embed: bsz * input_dim
        x = torch.stack(
            [self.router_units[i](task_embed) for i in range(self.all_layers)], dim=0
        )

        encoder_block_distribution = x[:self.encoder_layers]
        decoder_block_distribution = x[self.encoder_layers:]

        return encoder_block_distribution, decoder_block_distribution

class LinearRouter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder_layers = config.encoder_layers
        self.decoder_layers = config.decoder_layers
        self.n_blocks = config.router_block_num
        self.all_layers = config.encoder_layers + config.decoder_layers

        self.linear1 = nn.Linear(config.router_input_dim, config.router_hidden_dim)
        self.linear2 = nn.Linear(config.router_hidden_dim, self.n_blocks * self.all_layers)
        self.activation_fn = F.gelu

    def forward(self, task_embed):
        # task_embed: bsz * input_dim
        
        x = self.linear1(task_embed) # bsz * hidden_dim
        x = self.activation_fn(x)
        x = self.linear2(x) # bsz * (n_blocks x all_layers)

        x = x.view(-1, self.all_layers, self.n_blocks) # bsz * all_layers * n_blocks
        x = x.transpose(0,1) # all_layers * bsz * n_blocks

        encoder_block_distribution = x[:self.encoder_layers]
        decoder_block_distribution = x[self.encoder_layers:]

        return encoder_block_distribution, decoder_block_distribution

class RecurrentLayerSpecificRouter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder_layers = config.encoder_layers
        self.decoder_layers = config.decoder_layers
        self.all_layers = config.encoder_layers + config.decoder_layers

        # supposed to be (self.all_layers) linear layers of input_dim x hidden_dim
        self.layer_specific_proj = nn.Linear(config.router_input_dim, config.router_hidden_dim * self.all_layers)

        if config.router_nn_type == "rnn_layer_specific":
            self.rnn = nn.RNN(config.router_hidden_dim, config.router_hidden_dim, 
                bidirectional=True, batch_first=True)
        elif config.router_nn_type == "gru_layer_specific":
            self.rnn = nn.GRU(config.router_hidden_dim, config.router_hidden_dim, 
                bidirectional=True, batch_first=True)
        elif config.router_nn_type == "lstm_layer_specific":
            self.rnn = nn.LSTM(config.router_hidden_dim, config.router_hidden_dim, 
                bidirectional=True, batch_first=True)

        self.proj = nn.Linear(config.router_hidden_dim * 2, config.router_block_num) # *2 due to bidirectional

        self.incremental_indices = torch.arange(0, self.all_layers).long()
        if torch.cuda.is_available():
            self.incremental_indices = self.incremental_indices.cuda()

    def forward(self, task_embed):
        # task_embed: bsz * input_dim

        x = self.layer_specific_proj(task_embed) # bsz * (hidden_dim x all_layers)
        x = x.view(-1, self.all_layers, self.config.router_hidden_dim) # bsz * all_layers * hidden_dim
        x, hx = self.rnn(x)

        x = self.proj(x) # bsz * layer * router
        x = x.transpose(0, 1) # layer * bsz * router

        encoder_block_distribution = x[:self.encoder_layers, :, :]
        decoder_block_distribution = x[self.encoder_layers:, :, :]
        
        return encoder_block_distribution, decoder_block_distribution

class RecurrentRouter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder_layers = config.encoder_layers
        self.decoder_layers = config.decoder_layers
        self.all_layers = config.encoder_layers + config.decoder_layers

        self.pos_embed = nn.Embedding(self.all_layers, config.router_input_dim)

        if config.router_nn_type == "rnn":
            self.rnn = nn.RNN(config.router_input_dim, config.router_hidden_dim, 
                bidirectional=True, batch_first=True)
        elif config.router_nn_type == "gru":
            self.rnn = nn.GRU(config.router_input_dim, config.router_hidden_dim, 
                bidirectional=True, batch_first=True)
        elif config.router_nn_type == "lstm":
            self.rnn = nn.LSTM(config.router_input_dim, config.router_hidden_dim, 
                bidirectional=True, batch_first=True)

        self.proj = nn.Linear(config.router_hidden_dim * 2, config.router_block_num) # *2 due to bidirectional

        self.incremental_indices = torch.arange(0, self.all_layers).long()
        if torch.cuda.is_available():
            self.incremental_indices = self.incremental_indices.cuda()

    def forward(self, task_embed):
        # task_embed: bsz * input_dim

        task_embed = task_embed.expand(self.all_layers, -1, -1).transpose(0, 1) # bsz * all_layers * input_dim
        
        pos_input = self.incremental_indices.expand(task_embed.shape[0], self.all_layers)
        pos_embed = self.pos_embed(pos_input)

        x, hx = self.rnn(task_embed+pos_embed)

        x = self.proj(x) # bsz * layer * router
        x = x.transpose(0, 1) # layer * bsz * router

        encoder_block_distribution = x[:self.encoder_layers, :, :]
        decoder_block_distribution = x[self.encoder_layers:, :, :]
        
        return encoder_block_distribution, decoder_block_distribution


class TransformerLayerSpecificRouter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder_layers = config.encoder_layers
        self.decoder_layers = config.decoder_layers
        self.all_layers = config.encoder_layers + config.decoder_layers

        self.layer_specific_proj = nn.Linear(config.router_input_dim, config.router_hidden_dim * self.all_layers)

        self.transformer = nn.TransformerEncoderLayer(
            config.router_hidden_dim,
            nhead=4,
            dim_feedforward=64
            # batch_first=True # torch 1.7.1 doesn't have batch_first ...
        )

        self.proj = nn.Linear(config.router_hidden_dim, config.router_block_num)
        
        self.incremental_indices = torch.arange(0, self.all_layers).long()
        if torch.cuda.is_available():
            self.incremental_indices = self.incremental_indices.cuda()

    def forward(self, task_embed):
        # task_embed: bsz * input_dim

        x = self.layer_specific_proj(task_embed) # bsz * (hidden_dim x all_layers)
        x = x.view(-1, self.all_layers, self.config.router_hidden_dim) # bsz * all_layers * hidden_dim

        x = x.transpose(0, 1) # all_layers * bsz * hidden_dim
        x = self.transformer(x)

        x = self.proj(x) # bsz * layer * router

        encoder_block_distribution = x[:self.encoder_layers, :, :]
        decoder_block_distribution = x[self.encoder_layers:, :, :]
        
        return encoder_block_distribution, decoder_block_distribution

class TransformerRouter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder_layers = config.encoder_layers
        self.decoder_layers = config.decoder_layers
        self.all_layers = config.encoder_layers + config.decoder_layers

        self.pos_embed = nn.Embedding(self.all_layers, config.router_input_dim)

        self.transformer = nn.TransformerEncoderLayer(
            config.router_input_dim,
            nhead=4,
            dim_feedforward=64
            # batch_first=True # torch 1.7.1 doesn't have batch_first ...
        )

        self.proj = nn.Linear(config.router_input_dim, config.router_block_num)
        
        self.incremental_indices = torch.arange(0, self.all_layers).long()
        if torch.cuda.is_available():
            self.incremental_indices = self.incremental_indices.cuda()

    def forward(self, task_embed):
        # task_embed: bsz * input_dim

        task_embed = task_embed.expand(self.all_layers, -1, -1).transpose(0, 1) # bsz * all_layers * input_dim
        
        pos_input = self.incremental_indices.expand(task_embed.shape[0], self.all_layers)
        pos_embed = self.pos_embed(pos_input)

        x = task_embed+pos_embed # bsz * all_layers (i.e., seqlen) * hidden_dim
        x = x.transpose(0, 1) # all_layers * bsz * hidden_dim
        x = self.transformer(x)

        x = self.proj(x) # bsz * layer * router
        # x = x.transpose(0, 1) # layer * bsz * router

        encoder_block_distribution = x[:self.encoder_layers, :, :]
        decoder_block_distribution = x[self.encoder_layers:, :, :]
        
        return encoder_block_distribution, decoder_block_distribution