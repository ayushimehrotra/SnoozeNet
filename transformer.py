import torch.nn as nn
from utils import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding, FC_Softmax, clear_mem, attention

import gc
import numpy as np
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        '''
            Create one layer of an encoder 
            d_model: dimension of an input in a sequence
            num_heads: number of attention heads
            d_ff: intermediate dimension 
            dropout: parameter needed for dropout layer
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class EpochTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        '''
        d_model: dimension of the input in a sequence
        num_heads: number of attention heads
        num_layers: number of encoder layers
        d_ff: intermediate dimension
        max_seq_length: literally maximum sequence length
        dropout: parameter needed for dropout layer
        '''
        super(EpochTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads,
                                                          d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src_embedded = self.dropout(self.positional_encoding(src))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)

        del src_embedded
        clear_mem()
        return enc_output


class SequenceTransformer(nn.Module):
    def __init__(self, num_classes, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        '''
            num_classes: number of classes in the sleep stage classification
            d_model: input dimension at one step in sequence
            num_heads: number of heads in attention layer
            num_layers: number of encoder layers
            d_ff: intermediate dimension
            max_seq_length: maximum length of sequence
            dropout: parameter needed for dropout layer
        '''
        super(SequenceTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = FC_Softmax(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src_embedded = self.dropout(self.positional_encoding(src))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)

        output = self.fc(enc_output)
        return output


class SnoozeTransformer(nn.Module):
    def __init__(self, num_classes, d_model_e, num_heads_e, num_layers_e, d_ff_e,
                 d_model_s, num_heads_s, num_layers_s, d_ff_s, max_seq_length, max_seq_len_e, dropout):
        '''
        num_classes: number of classes for classification
        d_model_e: dimension of input at one point of sequence
        num_heads_e: number of heads in the attention for epoch transformer
        num_layers_e: number of encoder layers for epoch transformer
        d_ff_e: intermediate dimension for epoch transformer
        d_model_s: length of epoch
        num_heads_s: number of heads in the attention layber for sequence transformer
        d_ff_s: intermediate dimension for sequence transformer
        max_seq_length: maximum sequence length
        dropout: parameter needed for dropout layer
        '''
        super(SnoozeTransformer, self).__init__()

        self.epoch_transformers = nn.ModuleList([EpochTransformer(d_model_e, num_heads_e, num_layers_e, d_ff_e, max_seq_len_e, dropout) for _ in range(max_seq_length)])
        self.sequence_transformer = SequenceTransformer(num_classes, d_model_s, num_heads_s, num_layers_s, d_ff_s, max_seq_length, dropout)

    def forward(self, src):
        # src dimensions must be (batch_size, num_of_epoch, len_of_epoch, dimension_of_input) 
        transformed_epoch = []
        print(src.size())
        for i in range(src.size(1)):
            epoch_input = src[:, i, :, :]
            time_1 = time.time()
            epoch_output = self.epoch_transformers[i](epoch_input).cpu().detach().numpy()
            if len(transformed_epoch) % 100 == 0:
                print("[INFO] Processing EEG Epoch #" + str(len(transformed_epoch)))
                print(f"[INFO] Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
                time_2 = time.time()
                elapsed_time = time_2 - time_1
                print(f"Elapsed time: {elapsed_time} seconds")

            transformed_epoch.append(np.array(epoch_output))

            clear_mem()
        # (num_of_epochs, batch_size, len_of_epoch, dimension_of_input)
        transformed_epoch = np.transpose(np.array(transformed_epoch), (1, 0, 2, 3))
        # (batch_size, num_of_epochs, len_of_epoch, dimension_of_input)
        temp_transformed_output = []
        time_1 = time.time()
        for batch in transformed_epoch:
            temp_batch = []
            for epoch in batch:
                temp_epoch = []
                for d_model in epoch:
                    x = torch.tensor(np.expand_dims(d_model, axis=-1))
                    attn = attention(x, x, x)[0]
                    w = [i[0] for i in attn]
                    avg = np.average(d_model, weights=w)
                    temp_epoch.append(avg)
                temp_batch.append(temp_epoch) 
            temp_transformed_output.append(np.array(temp_batch, dtype=object))
            print("Batch Done")
        time_2 = time.time()
        elapsed_time = time_2 - time_1
        print(f"Elapsed time: {elapsed_time} seconds")
        transformed_epoch = np.array(temp_transformed_output, dtype=object)
        float_array = transformed_epoch.astype(np.float32)
        transformed_epoch = torch.tensor(float_array).cuda()
        
        output = self.sequence_transformer(transformed_epoch)
        return output