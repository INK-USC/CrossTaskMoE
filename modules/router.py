import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):

    # implementation reference: https://github.com/pytorch/fairseq/blob/main/examples/latent_depth/latent_depth_src/modules/latent_layers.py
    # a router ---
    # - takes in the meta-information about a task (e.g., task id, or fisher information)
    # - outputs a distribution over the blocks in each layer (e.g., 5 blocks to choose from in this layer, then the output is a 5d vector, and the sum will be 1)

    def __init__(self, input_dim, hidden_dim, block_num, mode, sampling_tau=5.0):
        super(Router, self).__init__()
        
        assert mode in ["softmax", "gumbel_softmax", "gumbel_softmax_st"]

        # self.register_backward_hook(lambda grad: grad * 100) # use 100x larger gradients for routers

        # let's use two layer perceptron for now, this can be any architecture
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, block_num)
        self.activation_fn = F.gelu # following what bart is using
        
        self.mode = mode
        self.tau = sampling_tau
        self.detach_grad = False

    def forward(self, x):
        # x contains some meta-information about the task
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)

        if self.mode == "softmax":
            x = F.softmax(x, dim=-1)
        elif self.mode == "gumbel_softmax":
            x = F.gumbel_softmax(x, tau=self.tau, hard=False)
        else: # gumbel_softmax_st
            x = F.gumbel_softmax(x, tau=self.tau, hard=True)

        return x

    def set_gumbel_temperature(self, tau):
        self.tau = tau