import torch
import math
import torch.nn as nn
import torch.nn.functional as F


# layer normalization of self-attention
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, dropout):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 2)

    def forward(self, input_tensor, epoch=0):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        out_dropout = nn.Dropout(0.8/(epoch/100+1))
        hidden_states = out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states+input_tensor)

        return hidden_states


class Clu_Head(nn.Module):
    def __init__(self, cfg, drop_out=-1, last_activation="softmax", batch_norm=False):
        super(Clu_Head, self).__init__()
        num_layer = len(cfg) - 1

        for i in range(num_layer-1):
            layer_name = "att{}".format(i+1)
            layer = SelfAttention(num_attention_heads=2, input_size=cfg[i], hidden_size=cfg[i], dropout=0.2)
            self.add_module(layer_name, layer)

            layer_name = "lin{}".format(i+1)
            layer = nn.Linear(cfg[i], cfg[i+1])
            self.add_module(layer_name, layer)

            if batch_norm:
                layer_name = "bn{}".format(i+1)
                layer = nn.BatchNorm1d(cfg[i+1])
                self.add_module(layer_name, layer)
            if drop_out > 0:
                layer_name = "drop{}".format(i + 1)
                layer = nn.Dropout(drop_out)
                self.add_module(layer_name, layer)
            layer_name = "relu{}".format(i + 1)
            layer = nn.ReLU(cfg[i + 1])
            self.add_module(layer_name, layer)

        layer_name = "att_final".format(i + 1)
        layer = SelfAttention(num_attention_heads=2, input_size=cfg[-2], hidden_size=cfg[-2], dropout=0.0)
        self.add_module(layer_name, layer)

        layer_name = "lin_final".format(i + 1)
        layer = nn.Linear(cfg[-2], cfg[-1])
        self.add_module(layer_name, layer)

        self.num_layer = num_layer
        self.drop_out = drop_out
        self.last_activation = last_activation
        self.batch_norm = batch_norm

    def forward(self, x, epoch = 0):
        num_layer = self.num_layer

        for i in range(num_layer-1):
            layer_name = "att{}".format(i+1)
            layer = self.__getattr__(layer_name)
            x = layer(x, epoch=epoch)

            layer_name = "lin{}".format(i+1)
            layer = self.__getattr__(layer_name)
            x = layer(x)

            if self.batch_norm:
                bn_name = "bn{}".format(i+1)
                bn = self.__getattr__(bn_name)
                x = bn(x)

            if self.drop_out > 0:
                drop_name = "drop{}".format(i + 1)
                drop = self.__getattr__(drop_name)
                x = drop(x)

            x = F.relu(x)

        layer_name = "lin_final".format(i + 1)
        layer = self.__getattr__(layer_name)
        x = layer(x)

        if self.last_activation == "relu":
            x = F.relu(x, inplace=True)
        elif self.last_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.last_activation == "exp_norm":
            x = torch.exp(x - x.max(dim=1)[0].unsqueeze(1))
        elif self.last_activation == "tanh":
            x = torch.tanh(x)
        elif self.last_activation == "softmax":
            softmax = torch.nn.Softmax(dim=1)
            x = softmax(x)
        elif self.last_activation is None:
            pass
        else:
            assert TypeError

        return x
