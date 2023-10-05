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


class ConvDecoder(nn.Module):
    def __init__(self, Rworld_shape, hidden_dim):
        super().__init__()
        self.RRworld_shape = tuple(map(lambda x: x // 32, Rworld_shape))
        self.linears = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(),
                                     nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(),
                                     nn.Linear(hidden_dim * 4, hidden_dim // 8 * math.prod(self.RRworld_shape)))
        self.deconvs = nn.Sequential(nn.Upsample(tuple(map(lambda x: x // 16, Rworld_shape)), mode='bilinear'),
                                     nn.Conv2d(hidden_dim // 8, hidden_dim // 8, 3, 1, 1), nn.ReLU(),
                                     nn.Upsample(tuple(map(lambda x: x // 8, Rworld_shape)), mode='bilinear'),
                                     nn.Conv2d(hidden_dim // 8, hidden_dim // 8, 3, 1, 1), nn.ReLU())
        self.covermap_head = nn.Conv2d(hidden_dim // 8, 1, 1)
        self.heatmap_head = nn.Conv2d(hidden_dim // 8, 1, 1)

    def forward(self, x):
        B, C = x.shape
        H, W = self.RRworld_shape
        x = self.linears(x).reshape([B, C // 8, H, W])
        x = self.deconvs(x)
        covermap = self.covermap_head(x)
        heatmap = self.heatmap_head(x)
        return covermap, heatmap


class CamControl(nn.Module):
    def __init__(self, dataset, hidden_dim, actstd_init=1.0):
        super().__init__()
        self.RRworld_shape = tuple(map(lambda x: x // 32, dataset.Rworld_shape))
        self.feat_branch = nn.Sequential(nn.Conv2d(1, hidden_dim, 3, 2, 1), nn.ReLU(),
                                         nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.ReLU(),
                                         nn.Conv2d(hidden_dim, hidden_dim // 8, 3, 2, 1),
                                         nn.AdaptiveMaxPool2d(self.RRworld_shape), nn.Flatten(),
                                         nn.Linear(hidden_dim // 8 * math.prod(self.RRworld_shape), hidden_dim * 4),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(),
                                         nn.Linear(hidden_dim * 4, hidden_dim))
        self.config_branch = nn.Sequential(nn.Linear(dataset.config_dim, hidden_dim), nn.LeakyReLU(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                           nn.Linear(hidden_dim, hidden_dim))
        self.feat_decoder = ConvDecoder(dataset.Rworld_shape, hidden_dim)

        # transformer
        # self.positional_embedding = create_pos_embedding(dataset.num_cam, hidden_dim)
        self.padding_token = nn.Parameter(torch.randn(hidden_dim))
        # CHECK: batch_first=True for transformer
        # NOTE: by default nn.Transformer() has enable_nested_tensor=True in its nn.TransformerEncoder(),
        # which can cause `src` to change from [B, 4, C] into [B, 1<=n<=3, C] when given `src_key_padding_mask`,
        # raising error for nn.TransformerDecoder()
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, 3)
        self.state_token = nn.Parameter(torch.randn([dataset.num_cam, hidden_dim]))

        # agent
        self.critic = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                    layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                        layer_init(nn.Linear(hidden_dim, len(dataset.action_names)), std=0.01))
        self.actor_std = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                       layer_init(nn.Linear(hidden_dim, len(dataset.action_names)), std=0.01))
        # self.actor_std = nn.Parameter(torch.zeros([len(dataset.action_names)]))

        self.actstd_init = actstd_init

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None, deterministic=False):
        heatmaps, configs, world_heatmap, step = state
        B, N, _ = configs.shape

        # transformer
        # x = x_feat = self.feat_branch(heatmaps.flatten(0, 1)[:, None]).unflatten(0, [B, N])
        # x_feat = self.feat_branch(heatmaps.max(dim=1, keepdim=True)[0])
        x_config = self.config_branch(configs.flatten(0, 1)).unflatten(0, [B, N])
        masked_location = torch.arange(N).repeat([B, 1]) > step[:, None]
        x_config[masked_location] = self.padding_token
        x = x_config
        token_location = (torch.arange(N).repeat([B, 1]) == step[:, None])
        # x_feat = self.feat_branch(heatmaps.flatten(0, 1)[:, None]).unflatten(0, [B, N])
        # x_config = self.config_branch(configs.flatten(0, 1).to(heatmaps.device)).unflatten(0, [B, N])
        # x = torch.cat([x_config, x_feat], dim=-1)
        # token_location = (torch.arange(N).repeat([B, 1]) == step[:, None])
        # query_feat = self.feat_branch(world_heatmap)
        # x[token_location] = torch.cat([self.state_token[step], query_feat], dim=-1)
        x[token_location] = self.state_token[step]
        x = F.layer_norm(x, [x.shape[-1]])  # + self.positional_embedding.to(x.device)
        # CHECK: batch_first=True for transformer
        x = self.transformer(x)
        x = x[token_location]

        # output head
        action_mean = self.actor_mean(x)
        action_std = F.softplus(torch.clamp(self.actor_std(x) + np.log(np.exp(self.actstd_init) - 1), -5, None))
        if deterministic:
            # remove randomness during evaluation when deterministic=True
            return action_mean, self.critic(x), None, None
        probs = Normal(action_mean, action_std)
        # http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        # alpha, beta = F.softplus(action_param[:, 0::2] - 3) + 1, F.softplus(action_param[:, 1::2] - 3) + 1
        # probs = Beta(alpha, beta)
        if action is None:
            action = probs.sample()
        return action, self.critic(x), probs, None


if __name__ == '__main__':
    import json
    import tqdm
    from torchvision.models import vit_b_16
    from torchvision.models import vgg16, alexnet
    from src.utils.tensor_utils import dist_action, dist_l2, dist_angle, expectation, tanh_prime
    from src.environment.carla_gym_seq import encode_camera_cfg


    class Object(object):
        pass


    B, N, C, H, W = 32, 4, 128, 200, 200
    dataset = Object()
    dataset.interactive = True
    dataset.Rworld_shape = [H, W]
    dataset.config_dim = 7
    dataset.action_names = ['x', 'y', 'z', 'pitch', 'yaw']
    dataset.num_cam = 4

    xy1, xy2 = torch.randn([7, 2]), torch.randn([10, 2])
    dist_loc = dist_l2(xy1[:, None], xy2[None])
    yaw1 = torch.tensor([0, 30, 45, 60, 90, 120, 180]) / 180
    yaw2 = torch.tensor([0, 15, 30, 60, 90, 150, 180, -120, -60, -180]) / 180
    dist_rot = dist_angle(yaw1[:, None], yaw2[None])

    with open('../../cfg/RL/1.cfg', "r") as fp:
        opts = json.load(fp)
    pos, dir = np.array(opts['cam_pos_lst']), np.array(opts['cam_dir_lst'])
    cam_configs_ = np.concatenate([pos, dir, np.ones([4, 1]) * opts['cam_fov']], axis=1)
    cam_configs = torch.tensor(np.array([encode_camera_cfg(cam_configs_[cam], opts) for cam in range(4)]))

    dist_action(cam_configs[None], cam_configs[:, None], dataset.action_names)

    model = CamControl(dataset, C, )
    # state_dict = torch.load(
    #     '../../logs/carlax/RL_1_6dof+fix_moda_E_lr0.0001_stdtanhinit0.5wait10_ent0.001_regdecay0.1e10_cover0.0_divsteps1.0mu0.0_dir0.0_det_TASK_max_e30_2023-10-02_22-51-37/model.pth')
    # state_dict = {key.replace('control_module.', ''): value for key, value in state_dict.items() if
    #               'control_module' in key}
    # model.load_state_dict(state_dict)
    # model.eval()
    heatmaps, configs, world_heatmap, step = (torch.randn([B, N, H, W]),
                                              torch.randn([B, N, dataset.config_dim]),
                                              torch.randn([B, 1, H, W]),
                                              torch.randint(0, N, [B]))
    masked_location = torch.arange(N).repeat([B, 1]) > step[:, None]
    heatmaps[masked_location] = -1
    configs[masked_location] = -3
    state = heatmaps, configs, world_heatmap, step
    action, value_, probs_, x_feat_ = model.get_action_and_value(state)
    action, value, probs, x_feat = model.get_action_and_value(state, action)
    # covermap, heatmap = model.feat_decoder(x_feat)

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
