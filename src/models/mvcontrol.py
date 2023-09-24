import time
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
from src.models.multiview_base import aggregate_feat, cover_mean, cover_mean_std


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
        self.feat_branch = nn.Sequential(nn.Conv2d(dataset.num_cam if arch == 'conv' else 1, hidden_dim, 3, 2, 1),
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
            # self.positional_embedding = nn.Parameter(torch.randn(dataset.num_cam + 1, hidden_dim))
            # CHECK: batch_first=True for transformer
            # NOTE: by default nn.Transformer() has enable_nested_tensor=True in its nn.TransformerEncoder(),
            # which can cause `src` to change from [B, 4, C] into [B, 1<=n<=3, C] when given `src_key_padding_mask`,
            # raising error for nn.TransformerDecoder()
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, 3)
            self.state_token = nn.Parameter(torch.randn(dataset.num_cam, hidden_dim))

        self.critic = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                                    layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.actor = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                                   layer_init(nn.Linear(hidden_dim, len(dataset.action_names) * 2), std=0.01))
        self.actstd_init = actstd_init

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None, deterministic=False):
        heatmaps, configs, step = state
        B, N, H, W = heatmaps.shape

        if self.arch == 'transformer':
            # transformer
            # x_feat = self.feat_branch(heatmaps)
            # x_config = self.config_branch(configs.flatten(0, 1).to(heatmaps.device)).unflatten(0, [B, N])
            # _, C = x_feat.shape
            # x = torch.cat([x_config, x_feat[:, None]], dim=1)
            # x = F.layer_norm(x, [C]) + self.positional_embedding
            # token_location = (torch.arange(N + 1).repeat([B, 1]) == step[:, None])
            # x[token_location] = F.layer_norm(self.state_token[step], [C])
            x_feat = self.feat_branch(heatmaps.flatten(0, 1)[:, None]).unflatten(0, [B, N])
            x_config = self.config_branch(configs.flatten(0, 1).to(heatmaps.device)).unflatten(0, [B, N])
            x = x_feat + x_config
            token_location = (torch.arange(N).repeat([B, 1]) == step[:, None])
            x[token_location] = self.state_token[step]
            x = F.layer_norm(x, [x.shape[-1]]) + self.positional_embedding.to(heatmaps.device)
            # CHECK: batch_first=True for transformer
            x = self.transformer(x)
            x = x[token_location]
        else:
            # conv + fc only
            x_feat = self.feat_branch(heatmaps)
            x_config = self.config_branch(configs.to(heatmaps.device).flatten(1, 2))
            x = x_feat + x_config

        # output head
        action_param = self.actor(x)
        action_mean = action_param[:, 0::2]
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
    from src.utils.tensor_utils import dist_action, dist_l2, dist_angle, expectation, tanh_prime


    class Object(object):
        pass


    B, N, C, H, W = 32, 4, 128, 200, 200
    dataset = Object()
    dataset.interactive = True
    dataset.config_dim = 7
    dataset.action_names = ['x', 'y', 'z', 'pitch', 'yaw']
    dataset.num_cam = 4

    xy1, xy2 = torch.randn([7, 2]), torch.randn([10, 2])
    dist_loc = dist_l2(xy1[:, None], xy2[None])
    yaw1 = torch.tensor([0, 30, 45, 60, 90, 120, 180]) / 180
    yaw2 = torch.tensor([0, 15, 30, 60, 90, 150, 180, -120, -60, -180]) / 180
    dist_rot = dist_angle(yaw1[:, None], yaw2[None])

    dist_action(torch.randn([7, 1, 5]), torch.randn([1, 10, 5]), dataset.action_names)

    model = CamControl(dataset, C, )
    # model.eval()

    state = (torch.randn([B, N, H, W]), torch.randn([B, N, dataset.config_dim]), torch.randint(0, N, [B]))
    model.get_action_and_value(state)

    # mu, sigma = torch.zeros([B, dataset.config_dim]), \
    #     torch.linspace(0.1, 10, B)[:, None].repeat([1, dataset.config_dim])
    # probs = Normal(mu, sigma)
    # z1 = expectation(probs, [probs.loc - 3 * probs.scale, probs.loc + 3 * probs.scale],
    #                  lambda x: -torch.stack([probs.log_prob(x[:, :, i]) for i in range(x.shape[-1])], dim=2))
    # t0 = time.time()
    # for _ in tqdm.tqdm(range(100)):
    #     z2 = expectation(probs, [probs.loc - 3 * probs.scale, probs.loc + 3 * probs.scale], tanh_prime)
    # print(time.time() - t0)

    # tgt = torch.randn([B, 1, C])
    # memory = torch.randn([B, N, C])
    # # transformer = nn.TransformerDecoderLayer(C, 8, C * 4, batch_first=True)
    # transformer = nn.Transformer(C, 8, 2, 2, C * 4, batch_first=True)
    # for _ in tqdm.tqdm(range(100)):
    #     transformer(tgt, memory)
