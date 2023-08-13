from typing import Optional, Any, Union, Callable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.nn.modules.transformer import _get_activation_fn
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


# remove the self attention layer for query input
class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                        **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                    **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            # x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            # x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    # def _sa_block(self, x: torch.Tensor,
    #               attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    #     x = self.self_attn(x, x, x,
    #                        attn_mask=attn_mask,
    #                        key_padding_mask=key_padding_mask,
    #                        need_weights=False)[0]
    #     return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: torch.Tensor, mem: torch.Tensor,
                   attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


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
            self.transformer = TransformerDecoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True)
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

    def get_action_and_value(self, state, action=None, deterministic=False):
        feat, configs, step = state
        B, N, C, H, W = feat.shape

        if self.arch == 'transformer':
            # transformer
            x_feat = self.feat_branch(feat.flatten(0, 1))[:, :, 0, 0].unflatten(0, [B, N])
            x_config = self.config_branch(configs.to(feat.device).flatten(0, 1)).unflatten(0, [B, N])
            # CHECK: batch_first=True for transformer
            x = x_feat + x_config + self.positional_embedding[None, :N]
            batch_state_token = self.state_token.expand(B, -1, -1).to(feat.device)
            padding_mask = (torch.arange(N).repeat([B, 1]) > step[:, None]).to(feat.device)
            # nn.MultiheadAttention requires *binary* mask=True for locations that we do not want to attend
            x = self.transformer(batch_state_token, x, memory_key_padding_mask=padding_mask)
            # Classifier "token" as used by standard language architectures
            x = x[:, 0]
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
