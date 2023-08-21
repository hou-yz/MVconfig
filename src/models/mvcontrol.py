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


class CamControl(nn.Module):
    def __init__(self, dataset, hidden_dim, arch='transformer', actstd_init=1.0):
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
            self.positional_embedding = create_pos_embedding(dataset.num_cam, hidden_dim)
            # CHECK: batch_first=True for transformer
            self.transformer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True)
            self.state_token = nn.Parameter(torch.randn(hidden_dim))
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
        self.actor_logstd = nn.Parameter(torch.ones(1, dataset.action_dim) * np.log(actstd_init))

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None, deterministic=False):
        feat, configs, step = state
        B, N, C, H, W = feat.shape

        if self.arch == 'transformer':
            # transformer
            x_feat = self.feat_branch(feat.flatten(0, 1))[:, :, 0, 0].unflatten(0, [B, N])
            x_config = self.config_branch(configs.to(feat.device).flatten(0, 1)).unflatten(0, [B, N])
            # CHECK: batch_first=True for transformer
            token_location = (torch.arange(N).repeat([B, 1]) == step[:, None] + 1)
            query = x_feat + x_config
            query[token_location] = self.state_token.expand(B, -1).to(feat.device)
            query += self.positional_embedding[None, :N].to(feat.device)
            query_padding_mask = (torch.arange(N).repeat([B, 1]) > step[:, None] + 1).to(feat.device)
            # key = x_feat + x_config + self.positional_embedding[None, :N].to(feat.device)
            # key_padding_mask = (torch.arange(N).repeat([B, 1]) > step[:, None]).to(feat.device)
            # nn.MultiheadAttention requires *binary* mask=True for locations that we do not want to attend
            # x = self.transformer(query, key,
            #                      tgt_key_padding_mask=query_padding_mask,
            #                      memory_key_padding_mask=key_padding_mask)
            x = self.transformer(query, src_key_padding_mask=query_padding_mask)
            # Classifier "token" as used by standard language architectures
            x = x[token_location]
        else:
            # conv + fc only
            x_feat = self.feat_branch(feat.max(dim=1)[0])[:, :, 0, 0]
            x_config = self.config_branch(configs.to(feat.device).flatten(1, 2))
            x = x_feat + x_config

        # output head
        action_mean = self.actor_mean(x)
        if not self.training and deterministic:
            # remove randomness during evaluation when deterministic=True
            return action_mean, None, None, self.critic(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
