import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import copy
from transformer import SnoozeTransformer

num_classes = 5
d_model_e = 80
max_seq_len_e = 40
num_heads_e = 4
num_layers_e = 4
d_ff_e = 256
d_model_s = 40
num_heads_s = 4
num_layers_s = 4
d_ff_s = 216
max_seq_length = 100
dropout = 0.1
num_epochs = 100
top_k = 1000

def make_model():
    transformer = SnoozeTransformer(num_classes, d_model_e, num_heads_e,
                                        num_layers_e, d_ff_e, d_model_s,
                                        num_heads_s, num_layers_s, d_ff_s,
                                        max_seq_length, max_seq_len_e, dropout)

    transformer = transformer.cuda()
    for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return transformer
