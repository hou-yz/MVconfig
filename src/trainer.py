import io
import itertools
import random
import time
import copy
import os
import contextlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from kornia.geometry import warp_perspective
import matplotlib.pyplot as plt
from PIL import Image
from src.parameters import *
from src.loss import *
from src.evaluation.evaluate import evaluate, evaluateDetection_py
from src.environment.carla_gym_seq import encode_camera_cfg
from src.environment.cameras import action2proj_mat
from src.utils.decode import ctdet_decode, mvdet_decode
from src.utils.nms import nms
from src.utils.meters import AverageMeter
from src.utils.projection import project_2d_points
from src.utils.image_utils import add_heatmap_to_image, img_color_denormalize
from src.utils.tensor_utils import expectation, tanh_prime, dist_action, dist_l2


class PerspectiveTrainer(object):
    def __init__(self, model, control_module, logdir, writer, args, ):
        self.model = model
        self.agent = control_module
        self.args = args
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.rl_global_step = 0
        self.writer = writer
        self.start_time = time.time()
        self.memory_bank = {'obs': [],
                            'actions': [],
                            'logprobs': [],
                            'rewards': [],
                            'dones': [],
                            'values': [],
                            'returns': [],
                            'advantages': [],
                            'world_gt': [],
                            'moda': [],
                            }
        # beta distribution in agent will output [0, 1], whereas the env has action space of [-1, 1]
        # self.action_mapping = lambda x: torch.clamp(x, -1, 1)
        self.action_mapping = torch.tanh

        # filter out visible locations
        xx, yy = np.meshgrid(np.arange(0, model.Rworld_shape[1]), np.arange(0, model.Rworld_shape[0]))
        self.unit_world_grids = torch.tensor(np.stack([xx, yy], axis=2), dtype=torch.float).flatten(0, 1)

    # https://github.com/vwxyzjn/ppo-implementation-details
    def expand_episode(self, dataset, init_obs, training=False, visualize=False, batch_idx=0):
        step, configs, _, _, _ = init_obs
        B, N, _ = configs.shape
        assert B == 1, 'currently only support batch size of 1 for the envs'
        # step 0: initialization
        next_done = False
        # for all N steps
        obs_heatmaps = torch.ones([B, N, *dataset.Rworld_shape]).cuda() * HEATMAP_PAD_VALUE  # key
        obs_covermaps = torch.zeros([B, N, *dataset.Rworld_shape], dtype=torch.bool).cuda()
        model_feat = torch.zeros([B, N, self.model.base_dim, *dataset.Rworld_shape]).cuda()
        world_heatmaps, world_offsets = [], []
        world_heatmap = torch.zeros([B, 1, *dataset.Rworld_shape]).cuda()  # query
        imgs = []
        action_history = []
        # for one single step
        cam_heatmap = torch.ones([B, 1, *dataset.Rworld_shape]).cuda() * HEATMAP_PAD_VALUE
        cam_covermap = torch.zeros([B, 1, *dataset.Rworld_shape], dtype=torch.bool).cuda()
        _, N, C, H, W = model_feat.shape
        while not next_done:
            # step 0 ~ N-1: action
            with torch.no_grad():
                action, value, probs, _ = self.agent.get_action_and_value(
                    (obs_heatmaps, configs.cuda(), world_heatmap, step),
                    deterministic=self.args.rl_deterministic and not training)
            if training:
                self.rl_global_step += 1 * B
                # Markovian if have (obs_heatmaps, configs, step) as state
                # only store cam_heatmap (one cam) instead of obs_heatmaps (all cams) to save memory
                self.memory_bank['obs'].append((cam_heatmap.cpu() if step.item() != 0 else None,
                                                cam_covermap.cpu() if step.item() != 0 else None,
                                                configs,
                                                world_heatmap,
                                                step))
                self.memory_bank['actions'].append(action.cpu())
                self.memory_bank['values'].append(value.item())
                self.memory_bank['logprobs'].append(probs.log_prob(action).sum(-1).item())

            (step, configs, img, aug_mat, proj_mat, world_gt, img_gt, frame), next_done = \
                dataset.step(self.action_mapping(action[0]).cpu())
            if training:
                self.memory_bank['dones'].append(next_done)
            imgs.append(img)
            action_history.append(action.cpu())
            step, configs = torch.tensor(step)[None], torch.tensor(configs, dtype=torch.float32)[None, :]
            visible_mask = project_2d_points(torch.inverse(proj_mat[0]).cuda(),
                                             self.unit_world_grids.cuda(),
                                             check_visible=True)[1].view([1, H, W])
            cam = step - 1
            with contextlib.nullcontext():  # torch.no_grad() if not next_done else
                feat, _ = self.model.get_feat(img[None, :].cuda(), aug_mat[None, :], proj_mat[None, :])
                feat *= visible_mask
                model_feat[:, cam] = feat
                world_heatmap, world_offset = self.model.get_output(model_feat[:, :step])
                cam_heatmap = self.model.get_world_heatmap(feat.flatten(0, 1))[:, 0].unflatten(0, [B, -1])
            # return them to train MVDet
            world_heatmaps.append(world_heatmap)
            world_offsets.append(world_offset)
            cam_covermap = feat.norm(dim=2).detach() != 0
            cam_heatmap = torch.sigmoid(cam_heatmap.detach()) * cam_covermap + HEATMAP_PAD_VALUE * ~cam_covermap
            world_heatmap = torch.sigmoid(world_heatmap.detach())
            obs_heatmaps[:, cam] = cam_heatmap
            obs_covermaps[:, cam] = cam_covermap

        # visualize
        def cover_visualize():
            avg_covermap = model_feat[0].norm(dim=1).bool().float().mean([0]).cpu()
            pedestrian_gt_ij = torch.where(world_gt['heatmap'][0] == 1)
            H, W = world_gt['heatmap'].shape[-2:]
            pedestrian_gt_ij = (world_gt['idx'][world_gt['reg_mask']] // W, world_gt['idx'][world_gt['reg_mask']] % W)
            fig = plt.figure(figsize=tuple(np.array(dataset.Rworld_shape)[::-1] / 50))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ax.imshow(avg_covermap + torch.sigmoid(world_heatmaps[-1].detach().cpu())[0, 0], vmin=0, vmax=2)
            ax.scatter(pedestrian_gt_ij[1], pedestrian_gt_ij[0], 4, 'orange', alpha=0.7)
            # https://stackoverflow.com/a/7821917/8305276
            # If we haven't already shown or saved the plot, then we need to
            # draw the figure first...
            fig.canvas.draw()

            # Now we can save it to a numpy array.
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            plt.cla()
            plt.clf()
            return data

        # step N (step range is 0 ~ N-1 so this is after done=True): calculate rewards
        if not training:
            if visualize:
                Image.fromarray(cover_visualize()).save(f'{self.logdir}/cover_{batch_idx}.png')
                save_image(make_grid(torch.cat(imgs), normalize=True), f'{self.logdir}/imgs_{batch_idx}.png')
            return model_feat.cpu(), (world_heatmaps[-1].detach().cpu(), world_offsets[-1].detach().cpu())

        rewards, stats = self.rl_rewards(
            dataset, action_history, model_feat[0].cpu(), (world_heatmaps, world_offsets), world_gt, frame)
        coverages, task_loss, modas, min_dist = stats
        # fixed episode length of num_cam - 1 so no need for value bootstrap
        returns, advantages = np.zeros([N]), np.zeros([N])
        values = np.array(self.memory_bank['values'][-N:])
        lastgaelam = 0
        for t in range(-1, -(N + 1), -1):
            nextnonterminal = 1.0 - self.memory_bank['dones'][t]
            if t == -1:  # last item [-1]
                next_value = next_return = 0
            else:  # second and third last item [-2], [-3], ...
                next_value = values[t + 1]
                next_return = returns[t + 1]
            if self.args.gae:
                delta = rewards[t] + self.args.gamma * next_value * nextnonterminal - values[t]
                advantages[t] = lastgaelam = \
                    delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns[t] = advantages[t] + values[t]
            else:
                returns[t] = rewards[t] + self.args.gamma * nextnonterminal * next_return
                advantages[t] = returns[t] - values[t]
        for t in range(N):
            self.memory_bank['rewards'].append(float(rewards[t]))
            self.memory_bank['advantages'].append(float(advantages[t]))
            self.memory_bank['returns'].append(float(returns[t]))
            self.memory_bank['moda'].append(float(modas[t]))
        self.memory_bank['world_gt'].append(world_gt)

        if self.writer is not None:
            self.writer.add_scalar("charts/episodic_return", rewards.sum().item(), self.rl_global_step)
            self.writer.add_scalar("charts/episodic_length", step.item(), self.rl_global_step)
            self.writer.add_scalar("charts/coverage", coverages[-1].item(), self.rl_global_step)
            self.writer.add_scalar("charts/action_dist", min_dist.mean().item(), self.rl_global_step)
            self.writer.add_scalar("charts/loss", task_loss, self.rl_global_step)
            self.writer.add_scalar("charts/moda", modas[-1].item(), self.rl_global_step)
            if visualize:
                self.writer.add_image("images/coverage", cover_visualize(), self.rl_global_step, dataformats='HWC')
                # self.writer.add_image("images/imgs", make_grid(torch.cat(imgs), normalize=True),
                #                       self.rl_global_step, dataformats='CHW')

        return model_feat, (world_heatmaps[-1], world_offsets[-1])

    def expand_mean_actions(self, dataset, ):
        configs = torch.ones([1, dataset.num_cam, dataset.config_dim]).cuda() * CONFIGS_PADDING_VALUE
        action_means = []
        for cam in range(dataset.num_cam):
            # step 0 ~ N-1: action
            action, _, _, _ = self.agent.get_action_and_value((None, configs, None, torch.tensor([cam])),
                                                              deterministic=True)
            token_location = (torch.arange(dataset.num_cam).repeat([1, 1]) == cam).cuda()
            new_config = encode_camera_cfg(dataset.base.env.action(action[0].detach(), cam), dataset.base.env.opts)
            configs = new_config * token_location[..., None] + configs * ~token_location[..., None]
            action_means.append(action)
        return torch.cat(action_means)

    # https://github.com/vwxyzjn/ppo-implementation-details
    def rl_rewards(self, dataset, action_history, model_feat, world_res, world_gt, frame):
        # coverage
        # N + 1, H, W
        N = dataset.num_cam
        obs_covermaps_ = torch.cat([torch.zeros([1, *dataset.Rworld_shape]), model_feat.norm(dim=1).bool().float()])
        overall_coverages = torch.stack([obs_covermaps_[:cam + 1].max(dim=0)[0].mean() for cam in range(N + 1)])
        weighted_cover_map = torch.stack(
            [torch.tanh(obs_covermaps_[:cam + 1].sum(0)) - torch.tanh(obs_covermaps_[:cam].sum(0))
             for cam in range(1, N + 1)])
        # weighted_cover_map = obs_covermaps[1:]
        weighted_cover_map *= world_gt['heatmap']
        # compute loss & moda based on final result
        task_losses = torch.zeros([N + 1])
        modas = torch.zeros([N + 1])
        for cam in range(N):
            world_heatmap, world_offset = world_res[0][cam].detach().cpu(), world_res[1][cam].detach().cpu()
            # loss
            task_losses[cam + 1] = focal_loss(world_heatmap, world_gt['heatmap'][None, :])
            # MODA
            xys = mvdet_decode(torch.sigmoid(world_heatmap), world_offset, reduce=dataset.world_reduce).cpu()
            grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
            positions = grid_xy
            ids = scores[0].squeeze() > self.args.cls_thres
            pos, s = positions[0, ids], scores[0, ids, 0]
            ids, count = nms(pos, s, 20, np.inf)
            res = torch.cat([torch.ones([count, 1]) * frame, pos[ids[:count]]], dim=1)
            moda, modp, precision, recall, stats = evaluateDetection_py(res, dataset.get_gt_array([frame]), [frame])
            modas[cam + 1] = moda
        # diversity
        action_history = torch.cat(action_history)
        action_dist = dist_action(self.action_mapping(action_history[:, None]),
                                  self.action_mapping(action_history[None]),
                                  dataset.action_names,
                                  self.args.div_xy_coef,
                                  self.args.div_yaw_coef)
        min_dist = torch.zeros([N])
        for i in range(1, N):
            min_dist[i] += torch.min(action_dist[i, :i])
        # use coverage, loss, or MODA as reward
        rewards = torch.zeros([N])
        if 'maxcover' in self.args.reward:
            # rewards += (overall_coverages[1:] - overall_coverages[:-1]) * 0.1
            rewards[-1] += overall_coverages[-1] * 0.1  # final step
        if 'avgcover' in self.args.reward:  # dense
            rewards += obs_covermaps_.mean(dim=[1, 2])[1:] * 0.1  # dense
        if 'weightcover' in self.args.reward:
            rewards += weighted_cover_map.sum(dim=[1, 2]) / (world_gt['heatmap'].sum() + 1e-8)
        if 'loss' in self.args.reward:
            rewards += (-task_losses[1:] + task_losses[:-1])  # dense
            # rewards[-1] += -task_losses[-1]  # final step
        if 'moda' in self.args.reward:
            # rewards += (modas[1:] - modas[:-1]) / 100
            rewards[-1] += modas[-1] / 100  # final step
        # encourage each action to be more dis-similar
        if 'div' in self.args.reward:
            rewards += min_dist * 0.01

        return rewards, (overall_coverages, task_losses[-1].item(), modas[1:], min_dist)

    def train_rl(self, dataset, optimizer, epoch_):
        # flatten the batch
        b_cam_heatmap, b_cam_covermap, b_configs, b_world_heatmap, b_step = [], [], [], [], []
        for (cam_heatmap, cam_covermap, configs, world_heatmap, step) in self.memory_bank['obs']:
            if cam_heatmap is not None:
                b_cam_heatmap.append(cam_heatmap)  # ppo_steps / dataset.num_cam * (dataset.num_cam - 1)
                b_cam_covermap.append(cam_covermap)  # ppo_steps / dataset.num_cam * (dataset.num_cam - 1)
            b_configs.append(configs)  # ppo_steps
            b_world_heatmap.append(world_heatmap)  # ppo_steps
            b_step.append(step)  # ppo_steps
        b_cam_heatmap, b_configs, b_step = torch.cat(b_cam_heatmap)[:, 0], torch.cat(b_configs), torch.cat(b_step)
        b_cam_covermap = torch.cat(b_cam_covermap)[:, 0]
        b_world_heatmap = torch.cat(b_world_heatmap)
        b_actions = torch.cat(self.memory_bank['actions'])
        b_logprobs = torch.tensor(self.memory_bank['logprobs'])
        b_rewards = torch.tensor(self.memory_bank['rewards'])
        b_dones = torch.tensor(self.memory_bank['dones'], dtype=torch.float)
        b_advantages = torch.tensor(self.memory_bank['advantages'])
        b_returns = torch.tensor(self.memory_bank['returns'])
        b_values = torch.tensor(self.memory_bank['values'])
        b_world_gts = {key: torch.stack([self.memory_bank['world_gt'][i][key]
                                         for i in range(len(self.memory_bank['world_gt']))])
                       for key in self.memory_bank['world_gt'][0].keys()}
        b_moda = torch.tensor(self.memory_bank['moda'])

        # reset memory bank
        self.memory_bank = {'obs': [],
                            'actions': [],
                            'logprobs': [],
                            'rewards': [],
                            'dones': [],
                            'values': [],
                            'returns': [],
                            'advantages': [],
                            'world_gt': [],
                            'moda': [],
                            }

        # only (L * (dataset.num_cam - 1) / dataset.num_cam) per-camera features are stored to save memory
        L, = b_step.shape
        _, N, _ = b_configs.shape
        _, H, W = b_cam_heatmap.shape
        B = self.args.rl_minibatch_size
        # append padding terms
        b_cam_heatmap = torch.cat([b_cam_heatmap, torch.ones([1, H, W]) * HEATMAP_PAD_VALUE])
        b_cam_covermap = torch.cat([b_cam_covermap, torch.zeros([1, H, W], dtype=torch.bool)])
        # idx for cam_heatmap
        idx_lookup = torch.stack([(b_step > 0)[:l + 1].sum() for l in range(L)]) - 1
        # where to find cam_heatmap indices in obs_heatmaps
        # if positive, then it should be all zeros for that cam_heatmap
        # if zero, it should be the same as idx_lookup
        # if negative, find previous locations
        idx_add_table = np.arange(N)[None, :] - np.arange(N)[:, None]
        idx_add_table = np.concatenate([np.ones([1, N], dtype=np.int64), idx_add_table])
        # if idx_add_table[b_step] <= 0, calculate the index for the non-zero cam_heatmap
        # else, the index should be -1
        b_heatmap_inds = (idx_lookup[:, None].repeat(1, N) + idx_add_table[b_step]) * (
                idx_add_table[b_step] <= 0) - (idx_add_table[b_step] > 0)
        # idx for action_history
        idx_add_table = np.arange(N)[None, :] - np.arange(N)[:, None]
        b_action_history_inds = torch.arange(L)[:, None].repeat([1, N]) + idx_add_table[b_step]
        b_world_gt_idx = torch.tensor([(b_step[:i + 1] == 0).sum().item() - 1 for i in range(L)])
        clipfracs = []
        # Optimizing the policy and value network
        for rl_epoch in range(self.args.rl_update_epochs):
            # re-calculate advantages
            self.agent.eval()
            b_advantages = torch.zeros_like(b_rewards)
            b_returns = torch.zeros_like(b_rewards)
            b_values = []
            b_inds = torch.arange(L)
            with torch.no_grad():
                for start in range(0, L, B):  # drop_last=False
                    end = min(start + B, L)
                    mb_inds = b_inds[start:end]
                    _, mb_values, _, _ = self.agent.get_action_and_value(
                        (b_cam_heatmap[b_heatmap_inds[mb_inds]].cuda(),
                         b_configs[mb_inds].cuda(),
                         b_world_heatmap[mb_inds],
                         b_step[mb_inds]),
                        b_actions[mb_inds].cuda())
                    b_values.append(mb_values.cpu())
                b_values = torch.cat(b_values).squeeze(1)
                # fixed episode length of num_cam - 1 so no need for value bootstrap
                if self.args.gae:
                    lastgaelam = 0
                    for t in reversed(range(L)):
                        nextnonterminal = 1.0 - b_dones[t]
                        nextvalues = 0 if t == L - 1 else b_values[t + 1]
                        delta = b_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - b_values[t]
                        b_advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * \
                                                       nextnonterminal * lastgaelam
                        b_returns[t] = b_advantages[t] + b_values[t]
                else:
                    for t in reversed(range(L)):
                        nextnonterminal = 1.0 - b_dones[t]
                        next_return = 0 if t == L - 1 else b_returns[t + 1]
                        b_returns[t] = b_rewards[t] + self.args.gamma * nextnonterminal * next_return
                        b_advantages[t] = b_returns[t] - b_values[t]

            # train agent
            self.agent.train()
            b_inds = torch.randperm(L)
            for start in range(0, L - B + 1, B):  # drop_last=True
                end = start + B
                mb_inds = b_inds[start:end]
                action, newvalue, probs, x_feat = self.agent.get_action_and_value(
                    (b_cam_heatmap[b_heatmap_inds[mb_inds]].cuda(),
                     b_configs[mb_inds].cuda(),
                     b_world_heatmap[mb_inds],
                     b_step[mb_inds]),
                    b_actions[mb_inds].cuda())
                newlogprob = probs.log_prob(action).sum(-1)
                logratio = newlogprob - b_logprobs[mb_inds].cuda()
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds].cuda()
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds].cuda()) ** 2
                    v_clipped = b_values[mb_inds].cuda() + torch.clamp(newvalue - b_values[mb_inds].cuda(),
                                                                       -self.args.clip_coef,
                                                                       self.args.clip_coef, )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds].cuda()) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds].cuda()) ** 2).mean()

                # https://arxiv.org/pdf/2006.05990.pdf
                # section B.8
                # entropy_loss = probs.entropy().sum(-1).mean()
                entropy_loss = (probs.entropy() +
                                expectation(probs, [probs.loc - 3 * probs.scale, probs.loc + 3 * probs.scale],
                                            tanh_prime, device='cuda')
                                ).sum(-1).mean()

                # div loss
                mb_action_history = b_actions[b_action_history_inds[mb_inds]].cuda()
                # action_dist = dist_l2(self.action_mapping(probs.loc[:, None]),
                #                       self.action_mapping(mb_action_history))
                action_dist, (xy, _, delta_xy, _) = dist_action(self.action_mapping(probs.loc[:, None]),
                                                                self.action_mapping(mb_action_history),
                                                                dataset.action_names,
                                                                self.args.div_xy_coef,
                                                                self.args.div_yaw_coef, True)
                steps_mask = torch.arange(N).repeat([B, 1]) < b_step[mb_inds, None]
                if steps_mask.any():
                    steps_div = torch.clamp(action_dist[steps_mask], 0, self.args.div_clamp).mean()
                else:
                    steps_div = torch.zeros([]).cuda()
                # make sure cameras won't look directly outside
                delta_dir = torch.clamp((xy + 0.1 * delta_xy).norm(dim=-1) - xy.norm(dim=-1), 0, None).mean() / 0.1
                # action diversity in terms of \mu
                mu_div_ = torch.zeros([N]).cuda()
                for step in range(1, N):  # skip the initial step as that one should be the same guess
                    idx = b_step[mb_inds] == step
                    if idx.sum().item() > 1:
                        mu_div_[step] = probs.loc[idx].std(dim=0).mean()
                mu_div = torch.clamp(mu_div_, 0, self.args.div_clamp).mean()
                # worldgrid(xy)_from_img(xy)
                action_mus = self.expand_mean_actions(dataset)
                mu_proj_mats = torch.stack([action2proj_mat(dataset, self.action_mapping(action_mus[cam]), cam)
                                            for cam in range(N)])
                mu_cover_map = warp_perspective(torch.ones([N, 1, *dataset.img_shape]).cuda(),
                                                mu_proj_mats, dataset.Rworld_shape)
                visible_masks = torch.stack([project_2d_points(torch.inverse(mu_proj_mats[cam]),
                                                               self.unit_world_grids.cuda(),
                                                               check_visible=True)[1].view([1, H, W])
                                             for cam in range(N)])
                mu_cover_map *= visible_masks
                mb_world_gts = {key: b_world_gts[key][b_world_gt_idx[mb_inds]] for key in b_world_gts.keys()}
                proj_cover = torch.tanh(mu_cover_map.sum(dim=0, keepdims=True)) * mb_world_gts['heatmap'].cuda()
                proj_cover = (proj_cover.sum([2, 3]) / (mb_world_gts['heatmap'].cuda().sum([2, 3]) + 1e-8)).mean()
                # proj_cover = -cover_loss(mu_proj_mats, mu_cover_map, history_cover_maps, mb_world_gts, dataset,
                #                        [self.args.cover_min_clamp, self.args.cover_max_clamp])
                recons_loss = torch.zeros([]).cuda()
                # covermap_recons, heatmap_recons = self.agent.feat_decoder(
                #     x_feat.detach() if self.args.autoencoder_detach else x_feat)
                # overall_covermap = F.interpolate(history_cover_maps.sum(dim=1, keepdim=True).float(),
                #                                  covermap_recons.shape[-2:]).bool()
                # overall_heatmap = F.interpolate(b_world_heatmap[mb_inds], covermap_recons.shape[-2:])
                # recons_loss = F.binary_cross_entropy(torch.sigmoid(covermap_recons), overall_covermap.float()) + \
                #               ((torch.sigmoid(heatmap_recons) - overall_heatmap).abs() * overall_covermap).mean()
                loss = (pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef) * self.args.use_ppo + \
                       (-steps_div * self.args.steps_div_coef - mu_div * self.args.mu_div_coef -
                        proj_cover * self.args.cover_coef + delta_dir * self.args.dir_coef +
                        recons_loss * self.args.autoencoder_coef
                        ) * self.args.reg_decay_factor ** ((epoch_ - 1) // self.args.reg_decay_epochs)

                if torch.isnan(loss):
                    print('**************** nan loss ****************')
                    print(pg_loss, v_loss, entropy_loss, steps_div, mu_div, proj_cover, delta_dir, recons_loss)
                if torch.isinf(loss):
                    print('**************** inf loss ****************')
                    print(pg_loss, v_loss, entropy_loss, steps_div, mu_div, proj_cover, delta_dir, recons_loss)
                # fix the action_std for the initial epochs
                optimizer.param_groups[-1]['lr'] = (epoch_ > self.args.std_wait_epochs) * \
                                                   self.args.control_lr * self.args.std_lr_ratio
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                optimizer.step()

            if self.args.target_kl is not None:
                if approx_kl > self.args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if self.writer is not None:
            self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[-1]["lr"], self.rl_global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.rl_global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.rl_global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.rl_global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.rl_global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.rl_global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.rl_global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.rl_global_step)
            self.writer.add_scalar("losses/steps_div", steps_div.item(), self.rl_global_step)
            self.writer.add_scalar("losses/mu_div", mu_div.item(), self.rl_global_step)
            self.writer.add_scalar("losses/delta_dir", delta_dir.item(), self.rl_global_step)
            self.writer.add_scalar("losses/ppl_cover", proj_cover.item(), self.rl_global_step)
            self.writer.add_scalar("losses/recons_loss", recons_loss.item(), self.rl_global_step)

        with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
            # action_space = dataset.base.env.opts['env_action_space'].split('-')
            print(f'v loss: {v_loss.item():.3f}, p loss: {pg_loss.item():.3f}, ent: {entropy_loss.item():.3f}, '
                  f'avg return: {b_returns.mean().item():.3f}\n'
                  f'sigma: {probs.scale.detach().cpu().mean(dim=0).numpy()}\n'
                  f'steps div: {steps_div.item():.3f}, mu div: {mu_div.item():.3f}, delta dir: {delta_dir.item():.3f}, '
                  f'ppl cover: {proj_cover.item():.3f}, recons loss: {recons_loss.item():.3f}')
            # if torch.where(b_step[mb_inds] == 0)[0].numel():
            #     idx = torch.where(b_step[mb_inds] == 0)[0][0].item()
            #     mu = probs.loc.detach().cpu().numpy()
            #     sigma = probs.scale.detach().cpu().numpy()
            #     # alpha = probs.concentration1.detach().cpu().numpy()
            #     # beta = probs.concentration0.detach().cpu().numpy()
            #     print(f'step 0: mu: \t{mu[idx]} \n        sigma: \t{sigma[idx]}'
            #           # f'step 0: \talpha: \t{alpha[idx]} \n        \tbeta: \t{beta[idx]}'
            #           )

        del b_cam_heatmap, b_configs, b_step, b_actions, b_logprobs, b_advantages, b_returns, b_values

    def train(self, epoch, dataloader, optimizers, schedulers=(), log_interval=100):
        losses = 0
        t0 = time.time()
        for batch_idx, (step, configs, imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) in enumerate(dataloader):
            B, N = configs.shape[:2]
            if self.args.other_lr_ratio == 0:
                self.model.eval()
                self.model.base.train()
            else:
                self.model.train()
            if self.args.base_lr_ratio == 0:
                self.model.base.eval()
            self.agent.eval()
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].flatten(0, 1)
            if self.args.interactive:
                feat, (world_heatmap, world_offset) = self.expand_episode(
                    dataloader.dataset, (step, configs, imgs, aug_mats, proj_mats),
                    True, visualize=self.rl_global_step % 500 == 0)
            else:
                (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = \
                    self.model(imgs.cuda(), aug_mats, proj_mats)
            # MVDet loss
            if self.args.use_mse:
                w_loss = F.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device))
            else:
                loss_w_hm = focal_loss(world_heatmap, world_gt['heatmap'])
                loss_w_off = regL1loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
                # loss_w_id = self.ce_loss(world_id, world_gt['reg_mask'], world_gt['idx'], world_gt['pid'])
                w_loss = loss_w_hm + loss_w_off  # + self.args.id_ratio * loss_w_id
            if not self.args.interactive:
                if self.args.use_mse:
                    img_loss = F.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device))
                else:
                    loss_img_hm = focal_loss(imgs_heatmap, imgs_gt['heatmap'])
                    loss_img_off = regL1loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
                    loss_img_wh = regL1loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])
                    # loss_img_id = self.ce_loss(imgs_id, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['pid'])
                    img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1  # + self.args.id_ratio * loss_img_id

            loss = w_loss + img_loss / N * self.args.alpha if not self.args.interactive else w_loss
            losses += loss.item()

            # train MVDet
            optimizers[0].zero_grad()
            loss.backward()
            optimizers[0].step()

            # train MVcontrol
            if len(self.memory_bank['obs']) >= self.args.ppo_steps:  # or batch_idx + 1 == len(dataloader)
                self.train_rl(dataloader.dataset, optimizers[1], epoch)

            for scheduler in schedulers:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))
            # logging
            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(dataloader):
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train epoch: {epoch}, batch:{(batch_idx + 1)}, '
                      f'loss: {losses / (batch_idx + 1):.3f}, time: {t_epoch:.1f}')
        for scheduler in schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()
        return losses / len(dataloader), None

    def test(self, dataloader):
        self.model.eval()
        self.agent.eval()
        t0 = time.time()
        losses = 0
        cover_avg = 0
        res_list = []
        for batch_idx, (step, configs, imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) in enumerate(dataloader):
            B, N = configs.shape[:2]
            with torch.no_grad():
                if self.args.interactive:
                    assert B == 1, 'only support batch_size/num_envs == 1'
                    feat, (world_heatmap, world_offset) = self.expand_episode(
                        dataloader.dataset, (step, configs, imgs, aug_mats, proj_mats),
                        visualize=(batch_idx < 5), batch_idx=batch_idx)
                else:
                    feat, _ = self.model.get_feat(imgs.cuda(), aug_mats, proj_mats)
                    world_heatmap, world_offset = self.model.get_output(feat.cuda())
                # coverage
                cam_coverages = feat.norm(dim=2).bool().float()
                overall_coverages = cam_coverages.max(dim=1)[0].mean().item()
            cover_avg += overall_coverages
            loss = focal_loss(world_heatmap, world_gt['heatmap'])
            if self.args.use_mse:
                loss = F.mse_loss(world_heatmap, world_gt['heatmap'].cuda())
            losses += loss.item()

            xys = mvdet_decode(torch.sigmoid(world_heatmap), world_offset,
                               reduce=dataloader.dataset.world_reduce).cpu()
            grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
            positions = grid_xy
            for b in range(B):
                ids = scores[b].squeeze() > self.args.cls_thres
                pos, s = positions[b, ids], scores[b, ids, 0]
                ids, count = nms(pos, s, 20, np.inf)
                res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                res_list.append(res)

        res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        # np.savetxt(f'{self.logdir}/test.txt', res, '%d')
        moda, modp, precision, recall, stats = evaluateDetection_py(res,
                                                                    dataloader.dataset.get_gt_array(),
                                                                    dataloader.dataset.frames)
        print(f'Test, cover: {cover_avg / len(dataloader):.3f}, loss: {losses / len(dataloader):.6f}, '
              f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%, '
              f'time: {time.time() - t0:.1f}s')

        if self.writer is not None:
            self.writer.add_scalar("results/moda", moda, self.rl_global_step)

        return losses / len(dataloader), [moda, modp, precision, recall]
