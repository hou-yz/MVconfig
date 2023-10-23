import time
import math
from kornia.geometry import warp_perspective
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
import torchvision.transforms as T
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.utils import make_grid, save_image
from src.models.multiview_base import aggregate_feat, cover_mean, cover_mean_std
from src.utils.image_utils import img_color_denormalize, array2heatmap
from src.utils.tensor_utils import to_tensor
from src.utils.projection import project_2d_points


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
    def __init__(self, dataset, hidden_dim, actstd_init=1.0, arch='encoder'):
        super().__init__()
        self.world_reduce, self.img_reduce = 40, 120
        self.Rworld_shape = tuple(map(lambda x: x // self.world_reduce, dataset.worldgrid_shape))
        self.Rworldgrid_from_worldcoord = np.linalg.inv(dataset.base.worldcoord_from_worldgrid_mat @
                                                        np.diag([self.world_reduce, self.world_reduce, 1]))
        self.action_names = dataset.action_names
        self.arch = arch

        if self.arch == 'conv' or self.arch == 'transformer':
            # filter out visible locations
            xx, yy = np.meshgrid(np.arange(0, self.Rworld_shape[1]), np.arange(0, self.Rworld_shape[0]))
            self.unit_world_grids = torch.tensor(np.stack([xx, yy], axis=2), dtype=torch.float).flatten(0, 1)

            self.base = efficientnet_b0(weights='DEFAULT').features
            # replace last two stride=2 with dilation
            self.base[4][0].block[1][0].stride = (1, 1)
            self.base[4][0].block[1][0].padding = (2, 2)
            self.base[4][0].block[1][0].dilation = (2, 2)
            self.base[6][0].block[1][0].stride = (1, 1)
            self.base[6][0].block[1][0].padding = (8, 8)
            self.base[6][0].block[1][0].dilation = (4, 4)
            self.bottleneck = nn.Sequential(nn.Conv2d(1280, hidden_dim, 1), nn.ReLU())
            self.feat_branch = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim // 4, 3, 1, 1), nn.ReLU(),
                                             nn.Conv2d(hidden_dim // 4, 4, 3, 1, 1),
                                             nn.Flatten(),
                                             nn.Linear(4 * math.prod(self.Rworld_shape), hidden_dim * 4), nn.ReLU(),
                                             nn.Linear(hidden_dim * 4, hidden_dim))
        if self.arch == 'encoder' or self.arch == 'transformer':
            self.config_branch = nn.Sequential(nn.Linear(dataset.config_dim, hidden_dim), nn.ReLU(),
                                               nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                               nn.Linear(hidden_dim, hidden_dim))

        # transformer
        # self.positional_embedding = create_pos_embedding(dataset.num_cam, hidden_dim)
        self.padding_token = nn.Parameter(torch.randn([hidden_dim]))
        # CHECK: batch_first=True for transformer
        # NOTE: by default nn.Transformer() has enable_nested_tensor=True in its nn.TransformerEncoder(),
        # which can cause `src` to change from [B, 4, C] into [B, 1<=n<=3, C] when given `src_key_padding_mask`,
        # raising error for nn.TransformerDecoder()
        if self.arch == 'encoder':
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, 3)
        elif self.arch == 'transformer':
            self.transformer = nn.Transformer(hidden_dim, 8, 3, 3, hidden_dim * 4, batch_first=True)
        elif self.arch == 'conv':
            pass
        else:
            raise Exception
        self.cls_token = nn.Parameter(torch.randn([dataset.num_cam, hidden_dim]))

        # agent
        self.critic = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                                    layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                                        layer_init(nn.Linear(hidden_dim, len(self.action_names)), std=0.01))
        # self.actor_std = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
        #                                layer_init(nn.Linear(hidden_dim, len(self.action_names)), std=0.01))
        self.actor_std = nn.Parameter(torch.zeros([len(dataset.action_names)]))

        self.actstd_init = actstd_init

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None, deterministic=False, visualize=False):
        step, configs, imgs, aug_mats, proj_mats = state
        B, N, _, H, W = imgs.shape

        padding_location = torch.arange(N).repeat([B, 1]) >= step[:, None]
        step_location = (torch.arange(N).repeat([B, 1]) == step[:, None])

        # feature branch
        if self.arch == 'conv' or self.arch == 'transformer':
            imgs = imgs.flatten(0, 1)
            inverse_aug_mats = torch.inverse(aug_mats.view([B * N, 3, 3]))
            # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
            imgcoord_from_Rimggrid_mat = inverse_aug_mats @ \
                                         torch.diag(torch.tensor([self.img_reduce, self.img_reduce, 1])
                                                    ).unsqueeze(0).repeat(B * N, 1, 1).float()
            # [input arg] proj_mats is worldcoord_from_imgcoord
            proj_mats = to_tensor(self.Rworldgrid_from_worldcoord)[None] @ \
                        proj_mats[:, :N].flatten(0, 1) @ \
                        imgcoord_from_Rimggrid_mat

            visible_mask = project_2d_points(torch.inverse(proj_mats).to(imgs.device),
                                             self.unit_world_grids.to(imgs.device),
                                             check_visible=True)[1].view([B, N, *self.Rworld_shape])

            if visualize:
                denorm = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                proj_imgs = warp_perspective(F.interpolate(imgs, scale_factor=1 / 8), proj_mats.to(imgs.device),
                                             self.Rworld_shape).unflatten(0, [B, N]) * visible_mask[:, :, None]
                # for cam in range(N):
                #     visualize_img = T.ToPILImage()(denorm(imgs)[cam])
                #     plt.imshow(visualize_img)
                #     plt.show()
                #     visualize_img = T.ToPILImage()(denorm(proj_imgs.detach())[0, cam])
                #     plt.imshow(visualize_img)
                #     plt.show()
                plt.imshow(make_grid(denorm(imgs[:N].cpu())).permute([1, 2, 0]))
                plt.show()
                plt.imshow(make_grid(denorm(proj_imgs[0].detach().cpu())).permute([1, 2, 0]))
                plt.show()

            imgs_feat = self.base(imgs)
            imgs_feat = self.bottleneck(imgs_feat)
            world_feat = warp_perspective(imgs_feat, proj_mats.to(imgs.device), self.Rworld_shape).unflatten(0, [B, N])
            world_feat *= visible_mask[:, :, None]

            # if visualize:
            #     for cam in range(N):
            #         visualize_img = array2heatmap(torch.norm(imgs_feat[cam * B].detach(), dim=0).cpu())
            #         plt.imshow(visualize_img)
            #         plt.show()
            #         visualize_img = array2heatmap(torch.norm(world_feat[0, cam].detach(), dim=0).cpu())
            #         # visualize_img.save(f'../../imgs/projfeat{cam + 1}.png')
            #         plt.imshow(visualize_img)
            #         plt.show()

        # config branch
        if self.arch == 'encoder' or self.arch == 'transformer':
            x_config = self.config_branch(configs.flatten(0, 1)).unflatten(0, [B, N])
            x_config[padding_location] = self.padding_token  # tgt & query
            x_config[step_location] = self.cls_token[step]
            x_config = F.layer_norm(x_config, [x_config.shape[-1]])

        # transformer
        # CHECK: batch_first=True for transformer
        if self.arch == 'encoder':
            x = self.transformer(x_config)
            x = x[step_location]
        elif self.arch == 'transformer':
            x_feat = self.feat_branch(world_feat.flatten(0, 1)).unflatten(0, [B, N])
            x_feat[padding_location] = self.padding_token  # src & key
            x_feat = F.layer_norm(x_feat, [x_feat.shape[-1]])

            x = self.transformer(x_feat, x_config)
            x = x[step_location]
        elif self.arch == 'conv':
            x = self.feat_branch(world_feat.max(dim=1)[0])
        else:
            raise Exception

        # output head
        action_mean = self.actor_mean(x)
        action_std = F.softplus(torch.clamp(self.actor_std + np.log(np.exp(self.actstd_init) - 1), -5, None))
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
    from torch.utils.data import DataLoader
    from src.datasets import frameDataset, CarlaX
    from src.utils.tensor_utils import dist_action, dist_l2, dist_angle, expectation, tanh_prime


    class Object(object):
        pass


    B, N, C, H, W = 16, 4, 128, 200, 200
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

    with open('../../cfg/RL/town04crossroad.cfg', "r") as fp:
        dataset_config = json.load(fp)
    dataset = frameDataset(CarlaX(dataset_config, port=2000, tm_port=8000, euler2vec='yaw-pitch'), interactive=True,
                           seed=0)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    (step, configs, imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) = next(iter(dataloader))
    imgs = F.interpolate(imgs.flatten(0, 1), scale_factor=1 / 10).unflatten(0, [*configs.shape[:2]])

    model = CamControl(dataset, C, arch='transformer').cuda()
    # state_dict = torch.load(
    #     '../../logs/carlax/RL_1_6dof+fix_moda_E_lr0.0001_stdtanhinit0.5wait10_ent0.001_regdecay0.1e10_cover0.0_divsteps1.0mu0.0_dir0.0_det_TASK_max_e30_2023-10-02_22-51-37/model.pth')
    # state_dict = {key.replace('control_module.', ''): value for key, value in state_dict.items() if
    #               'control_module' in key}
    # model.load_state_dict(state_dict)
    # model.eval()
    # heatmaps, configs, world_heatmap, step = (torch.randn([B, N, H, W]),
    #                                           torch.randn([B, N, dataset.config_dim]),
    #                                           torch.randn([B, 1, H, W]),
    #                                           torch.randint(0, N, [B]))
    # masked_location = torch.arange(N).repeat([B, 1]) > step[:, None]
    # heatmaps[masked_location] = -1
    # configs[masked_location] = -3
    state = step.repeat(B), configs.repeat([B, 1, 1]).cuda(), imgs.repeat([B, 1, 1, 1, 1]).cuda(), \
        aug_mats.repeat([B, 1, 1, 1]), proj_mats.repeat([B, 1, 1, 1])
    action, value_, probs_, x_feat_ = model.get_action_and_value(state, visualize=True)
    action, value, probs, x_feat = model.get_action_and_value(state, action)
    (step, configs, img, aug_mat, proj_mat, world_gt, img_gt, frame), next_done = \
        dataset.step(action.cpu().numpy()[0], visualize=True)
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
