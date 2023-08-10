import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import math


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def create_pos_embedding(L, hidden_dim=128, temperature=10000, ):
    position = torch.arange(L).unsqueeze(1) / L * 2 * np.pi
    # div_term = temperature ** (torch.arange(0, hidden_dim, 2) / hidden_dim)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (math.log(temperature) / hidden_dim))
    pe = torch.zeros(L, hidden_dim)
    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)
    return pe


# torchvision/models/vision_transformer.py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CamControl(nn.Module):
    def __init__(self, dataset, hidden_dim, arch='transformer'):
        super().__init__()
        self.arch = arch
        if arch == 'transformer':
            # transformer
            self.feat_branch = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                                             nn.LeakyReLU(),
                                             nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                                             nn.LeakyReLU(),
                                             nn.AdaptiveAvgPool2d((1, 1)))
            self.config_branch = nn.Sequential(nn.Linear(dataset.config_dim, hidden_dim), nn.LeakyReLU(),
                                               nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), )
            positional_embedding = create_pos_embedding(dataset.num_cam, hidden_dim)
            self.register_buffer('positional_embedding', positional_embedding)
            # CHECK: batch_first=True for transformer
            self.transformer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True)
            self.state_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            # conv + fc only
            self.feat_branch = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                                             nn.LeakyReLU(),
                                             nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                                             nn.LeakyReLU(),
                                             nn.AdaptiveAvgPool2d((1, 1)))
            self.config_branch = nn.Sequential(nn.Linear(dataset.config_dim * dataset.num_cam, hidden_dim),
                                               nn.LeakyReLU(),
                                               nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), )

        self.critic = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                    layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                    layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                        layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                        layer_init(nn.Linear(hidden_dim, dataset.action_dim), std=0.01))
        self.actor_logstd = nn.Parameter(torch.ones(1, dataset.action_dim) * np.log(0.1))

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        feat, configs, step = state
        B, N, C, H, W = feat.shape
        padding_mask = torch.arange(N).repeat([B, 1]) > step[:, None]

        if self.arch == 'transformer':
            # transformer
            x_feat = self.feat_branch(feat.flatten(0, 1))[:, :, 0, 0].unflatten(0, [B, N])
            x_config = self.config_branch(configs.to(feat.device).flatten(0, 1)).unflatten(0, [B, N])
            # CHECK: batch_first=True for transformer
            x = x_feat + x_config + self.positional_embedding[None, :N]
            batch_state_token = self.state_token.expand(B, -1, -1)
            x = torch.cat([batch_state_token, x], dim=1)
            ext_padding_mask = torch.cat([torch.zeros([B, 1]), padding_mask], dim=1).bool().to(feat.device)
            # nn.MultiheadAttention requires *binary* mask=True for locations that we do not want to attend
            x = self.transformer(x, src_key_padding_mask=ext_padding_mask)
            # Classifier "token" as used by standard language architectures
            x = x[:, 0]
        else:
            # conv + fc only
            x_feat = self.feat_branch(feat.max(dim=1)[0])[:, :, 0, 0]
            x_config = self.config_branch(configs.to(feat.device).flatten(1, 2))
            x = x_feat + x_config

        # output head
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
