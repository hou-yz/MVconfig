import time
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta


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


def expectation(probs, x_range, func, n_points=1000, device='cpu'):
    if isinstance(x_range[0], float):
        B, C = 1, 1
        x_range = [torch.ones(B, C) * (x_range[0]), torch.ones(B, C) * (x_range[0])]
    elif len(x_range[0].shape) == 0:
        B, C = 1, 1
        x_range = [x_range[0].reshape([B, C]), x_range[0].reshape([B, C])]
    elif len(x_range[0].shape) == 1:
        B, C = 1, x_range.shape[0]
        x_range = [x_range[0].reshape([B, C]), x_range[0].reshape([B, C])]
    elif len(x_range[0].shape) == 2:
        B, C = x_range[0].shape
        pass
    else:
        raise Exception
    x = torch.empty([B, C, n_points], device=device)
    for b in range(B):
        for c in range(C):
            x[b, c] = torch.linspace(float(x_range[0][b, c]), float(x_range[1][b, c]), n_points, device=device)
    log_probs = torch.stack([probs.log_prob(x[:, :, i]) for i in range(n_points)], dim=2)
    return torch.trapz(torch.exp(log_probs) * func(x), x)


class CamControl(nn.Module):
    def __init__(self, dataset, hidden_dim, arch='transformer', actstd_init=1.0):
        super().__init__()
        self.arch = arch
        self.feat_branch = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                                         nn.ReLU(), nn.MaxPool2d(2, 2),
                                         nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                                         nn.ReLU(), nn.MaxPool2d(2, 2),
                                         nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                                         nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.config_branch = nn.Sequential(nn.Linear(dataset.config_dim * (dataset.num_cam if arch == 'conv'
                                                                           else 1), hidden_dim), nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim))
        if arch == 'transformer':
            # transformer
            self.positional_embedding = create_pos_embedding(dataset.num_cam, hidden_dim)
            # CHECK: batch_first=True for transformer
            # NOTE: by default nn.Transformer() has enable_nested_tensor=True in its nn.TransformerEncoder(),
            # which can cause `src` to change from [B, 4, C] into [B, 1<=n<=3, C] when given `src_key_padding_mask`,
            # raising error for nn.TransformerDecoder()
            self.transformer = nn.Transformer(hidden_dim, 8, 2, 2, hidden_dim * 4, batch_first=True)
            self.state_token = nn.Parameter(torch.randn(dataset.num_cam, hidden_dim))

        self.critic = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                                    layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                                    layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.actor = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                                   layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                                   layer_init(nn.Linear(hidden_dim, dataset.action_dim * 2), std=0.01))
        self.actstd_init = actstd_init

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None, deterministic=False):
        feat, configs, step = state
        B, N, C, H, W = feat.shape

        if self.arch == 'transformer':
            # transformer
            x_feat = self.feat_branch(feat.flatten(0, 1)).unflatten(0, [B, N])
            x_config = self.config_branch(configs.flatten(0, 1).to(feat.device)).unflatten(0, [B, N])
            x = x_feat + x_config + self.positional_embedding.to(feat.device)
            query = self.state_token[step, None]
            # CHECK: batch_first=True for transformer
            x = self.transformer(x, query)
            x = x[:, 0]
        else:
            # conv + fc only
            x_feat = self.feat_branch(feat.max(dim=1)[0])
            x_config = self.config_branch(configs.to(feat.device).flatten(1, 2))
            x = x_feat + x_config

        # output head
        action_param = self.actor(x)
        action_mean = torch.tanh(action_param[:, 0::2])
        action_std = F.softplus(action_param[:, 1::2] + np.log(np.exp(self.actstd_init) - 1))
        if not self.training and deterministic:
            # remove randomness during evaluation when deterministic=True
            return action_mean, self.critic(x), None
        probs = Normal(action_mean, action_std)
        # http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        # alpha, beta = F.softplus(action_param[:, 0::2] - 3) + 1, F.softplus(action_param[:, 1::2] - 3) + 1
        # probs = Beta(alpha, beta)
        if action is None:
            action = probs.sample()
        return action, self.critic(x), probs


if __name__ == '__main__':
    import tqdm
    from torchvision.models import vit_b_16
    from torchvision.models import vgg16, alexnet


    class Object(object):
        pass


    B, N, C, H, W = 32, 4, 128, 200, 200
    dataset = Object()
    dataset.config_dim = 7
    dataset.action_dim = 2
    dataset.num_cam = 4

    model = CamControl(dataset, C, )
    # model.eval()

    state = (torch.randn([B, N, C, H, W]), torch.randn([B, N, dataset.config_dim]), torch.randint(0, N - 1, [B]))
    t0 = time.time()
    for _ in tqdm.tqdm(range(10)):
        # with torch.no_grad():
        model.get_action_and_value(state)
    print(time.time() - t0)

    mu, sigma = torch.randn([B, dataset.config_dim]), torch.rand([B, dataset.config_dim])
    probs = Normal(mu, sigma)
    z1 = expectation(probs, [probs.loc - 3 * probs.scale, probs.loc + 3 * probs.scale],
                     lambda x: -torch.stack([probs.log_prob(x[:, :, i]) for i in range(x.shape[-1])], dim=2))
    t0 = time.time()
    for _ in tqdm.tqdm(range(100)):
        z2 = expectation(probs, [probs.loc - 3 * probs.scale, probs.loc + 3 * probs.scale],
                         lambda x: torch.log(4 / (torch.exp(x) + torch.exp(-x)) ** 2))
    print(time.time() - t0)

    # tgt = torch.randn([B, 1, C])
    # memory = torch.randn([B, N, C])
    # # transformer = nn.TransformerDecoderLayer(C, 8, C * 4, batch_first=True)
    # transformer = nn.Transformer(C, 8, 2, 2, C * 4, batch_first=True)
    # for _ in tqdm.tqdm(range(100)):
    #     transformer(tgt, memory)
