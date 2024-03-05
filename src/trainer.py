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
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from kornia.geometry import warp_perspective
import matplotlib.pyplot as plt
from PIL import Image
from src.parameters import *
from src.loss import *
from src.evaluation.evaluate import evaluate, evaluateDetection_py
from src.evaluation.mot_bev import mot_metrics_pedestrian
from src.environment.cameras import action2proj_mat
from src.tracking.multitracker import JDETracker
from src.utils.decode import ctdet_decode, mvdet_decode
from src.utils.nms import nms
from src.utils.meters import AverageMeter
from src.utils.projection import project_2d_points
from src.utils.image_utils import add_heatmap_to_image, img_color_denormalize
from src.utils.tensor_utils import expectation, tanh_prime, dist_action, dist_l2, to_tensor


# visualize
def cover_visualize(dataset, model_feat, world_heatmap, world_gt):
    N, C, H, W = model_feat.shape
    avg_covermap = model_feat.norm(dim=1).bool().float().mean([0]).cpu()
    pedestrian_gt_ij = torch.where(world_gt['heatmap'][0] == 1)
    pedestrian_gt_ij = (world_gt['idx'][world_gt['reg_mask']] // W,
                        world_gt['idx'][world_gt['reg_mask']] % W)
    fig = plt.figure(figsize=tuple(np.array(dataset.Rworld_shape)[::-1] / 50))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(avg_covermap + torch.sigmoid(world_heatmap[0].detach().cpu()), vmin=0, vmax=2)
    ax.scatter(pedestrian_gt_ij[1], pedestrian_gt_ij[0], 6, 'orange', alpha=0.7)
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
                            }
        # beta distribution in agent will output [0, 1], whereas the env has action space of [-1, 1]
        # self.action_mapping = lambda x: torch.clamp(x, -1, 1)
        # self.action_mapping = torch.tanh
        self.best_action = None
        self.best_episodic_return = 0

        # filter out visible locations
        xx, yy = np.meshgrid(np.arange(0, model.Rworld_shape[1]), np.arange(0, model.Rworld_shape[0]))
        # self.unit_world_grids = torch.tensor(np.stack([xx, yy], axis=2), dtype=torch.float).flatten(0, 1)
        self.unit_world_grids = F.interpolate(torch.tensor(np.stack([xx, yy]), dtype=torch.float)[None],
                                              scale_factor=1 / self.args.visibility_reduce)[0]
        self.unit_world_grids = torch.cat([self.unit_world_grids,
                                           torch.ones([1, *self.unit_world_grids.shape[1:]])]).flatten(1).cuda()

    def action_mapping(self, action):
        if self.args.action_mapping == 'clip':
            direction_idx = torch.tensor(['dir' in name for name in self.agent.action_names]).to(action.device)
            action = action / (action[..., direction_idx].norm(dim=-1, keepdim=True) * direction_idx +
                               ~direction_idx + 1e-8)
            action = torch.clamp(action, -1, 1)
        elif self.args.action_mapping == 'tanh':
            action = torch.tanh(action)
        else:
            raise Exception
        return action

    # https://github.com/vwxyzjn/ppo-implementation-details
    def expand_episode(self, dataset, init_obs, training=False, visualize=False, batch_idx=0, override_action=None):
        step, configs, imgs, aug_mats, proj_mats = init_obs
        B, N, _ = configs.shape
        imgs = F.interpolate(imgs.flatten(0, 1), scale_factor=1 / 10).unflatten(0, [B, N])
        assert B == 1, 'currently only support batch size of 1 for the envs'
        # step 0: initialization
        next_done = False
        cam = None
        # for all N steps
        action_history = []
        while not next_done:
            # step 0 ~ N-1: action
            if override_action is not None:
                action = override_action[step]
            else:
                with torch.no_grad():
                    action, value, probs, _ = self.agent.get_action_and_value(
                        (step, configs.cuda(), imgs.cuda(), aug_mats, proj_mats),
                        deterministic=self.args.rl_deterministic and not training)
            if training:
                self.rl_global_step += 1 * B
            if training and override_action is None:
                # Markovian if have (obs_heatmaps, configs, step) as state
                # only store cam_heatmap (one cam) instead of obs_heatmaps (all cams) to save memory
                self.memory_bank['obs'].append((step,
                                                configs,
                                                imgs[:, cam] if cam is not None else None,
                                                aug_mats,
                                                proj_mats))
                self.memory_bank['actions'].append(action.cpu())
                self.memory_bank['values'].append(value.item())
                self.memory_bank['logprobs'].append(probs.log_prob(action).sum(-1).item())

            (step, config, img, aug_mat, proj_mat, _, _, _), next_done = \
                dataset.step(self.action_mapping(action[0]).cpu())
            if training and override_action is None:
                self.memory_bank['dones'].append(next_done)
            cam = step - 1
            step = to_tensor(step, dtype=torch.long)[None]
            configs[:, cam], aug_mats[:, cam], proj_mats[:, cam] = config, aug_mat, proj_mat
            imgs[:, cam] = F.interpolate(img, scale_factor=1 / 10)
            action_history.append(action.cpu())
        # step N (step range is 0 ~ N-1 so this is after done=True): calculate rewards
        (step, configs, imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) = \
            dataset.__getitem__(use_depth=self.args.reward == 'visibility')
        imgs = to_tensor(imgs)[None]
        aug_mats, proj_mats = to_tensor(aug_mats)[None], to_tensor(proj_mats)[None]
        model_feat, _ = self.model.get_feat(imgs.cuda(), aug_mats, proj_mats)
        world_feat, (world_heatmap, world_offset, world_id) = self.model.get_output(model_feat.cuda())

        if not training:
            if visualize:
                Image.fromarray(cover_visualize(dataset, model_feat[0], world_heatmap[0], world_gt)
                                ).save(f'{self.logdir}/cover_{batch_idx}.png')
                save_image(make_grid(imgs[0], 6, normalize=True), f'{self.logdir}/imgs_{batch_idx}.png')
            return (model_feat.cpu(), (world_feat.detach().cpu(), world_heatmap.detach().cpu(),
                                       world_offset.detach().cpu(), world_id.detach().cpu()),
                    ({key: value[None] for key, value in world_gt.items()},
                     {key: value[None] for key, value in imgs_gt.items()}), torch.cat(action_history))

        # NOTE: Not adding tracking to reward computation since tracking task requires multiple frames
        #       to produce performance evaluation. However, at the end of each episode we only get a single frame.
        # TODO: But this feature may be implemented in the future - using track metrics as reward functions!!!
        rewards, stats = self.rl_rewards(dataset, action_history, model_feat[0].cpu(), (world_heatmap, world_offset),
                                         world_gt, frame, proj_mats, imgs_gt)
        coverages, task_loss, moda, min_dist = stats
        if rewards.sum().item() > self.best_episodic_return:
            self.best_episodic_return = rewards.sum().item()
            self.best_action = torch.cat(action_history)
        # fixed episode length of num_cam - 1 so no need for value bootstrap
        returns, advantages = np.zeros([N]), np.zeros([N])
        if override_action is None:
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
            self.memory_bank['world_gt'].append(world_gt)

        if self.writer is not None:
            self.writer.add_scalar("charts/episodic_return", rewards.sum().item(), self.rl_global_step)
            self.writer.add_scalar("charts/episodic_length", step, self.rl_global_step)
            self.writer.add_scalar("charts/coverage", coverages[-1].item(), self.rl_global_step)
            self.writer.add_scalar("charts/action_dist", min_dist.mean().item(), self.rl_global_step)
            self.writer.add_scalar("charts/loss", task_loss, self.rl_global_step)
            self.writer.add_scalar("charts/moda", moda, self.rl_global_step)
            if visualize:
                self.writer.add_image("images/coverage",
                                      cover_visualize(dataset, model_feat[0], world_heatmap[0], world_gt),
                                      self.rl_global_step, dataformats='HWC')
                # self.writer.add_image("images/imgs", make_grid(torch.cat(imgs), normalize=True),
                #                       self.rl_global_step, dataformats='CHW')

        return (model_feat, (world_feat, world_heatmap, world_offset, world_id),
                                ({key: value[None] for key, value in world_gt.items()},
                                 {key: value[None] for key, value in imgs_gt.items()}), torch.cat(action_history))

    # https://github.com/vwxyzjn/ppo-implementation-details
    def rl_rewards(self, dataset, action_history, model_feat, world_res, world_gt, frame, proj_mats, imgs_gt):
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
        world_heatmap, world_offset = world_res[0].detach().cpu(), world_res[1].detach().cpu()
        # loss
        task_loss = focal_loss(world_heatmap, world_gt['heatmap'][None, :])
        # MODA
        xys = mvdet_decode(torch.sigmoid(world_heatmap), world_offset, reduce=dataset.world_reduce).cpu()
        grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
        positions = grid_xy
        ids = scores[0].squeeze() > self.args.cls_thres
        pos, s = positions[0, ids], scores[0, ids, 0]
        ids, count = nms(pos, s, 20, np.inf)
        res = torch.cat([torch.ones([count, 1]) * frame, pos[ids[:count]]], dim=1)
        moda, modp, precision, recall, stats = evaluateDetection_py(res, dataset.get_gt_array([frame]), [frame])
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
            # rewards += (-task_losses[1:] + task_losses[:-1])  # dense
            rewards[-1] += -task_loss  # final step
        if 'moda' in self.args.reward:
            # rewards += (modas[1:] - modas[:-1]) / 100
            rewards[-1] += moda / 100  # final step
        # encourage each action to be more dis-similar
        if 'div' in self.args.reward:
            rewards += min_dist * 0.01
        if 'visibility' in self.args.reward:
            # visibility
            proj_mats_ = to_tensor(dataset.Rworldgrid_from_worldcoord)[None] @ proj_mats.flatten(0, 1)
            proj_imgs_points = (torch.inverse(proj_mats_.cuda()) @ self.unit_world_grids).permute([0, 2, 1]).cpu()
            proj_imgs_points[..., :2] /= proj_imgs_points[..., [2]]
            world_point_visibility = torch.zeros(proj_imgs_points.shape[:2])
            for cam in range(N):
                for i in range(self.unit_world_grids.shape[1]):
                    u, v, d = proj_imgs_points[cam, i]
                    if u > 0 and u < dataset.img_shape[1] and v > 0 and v < dataset.img_shape[0]:
                        u_, v_, d_ = int(u), int(v), d.item()
                        d_img = imgs_gt['depth'][cam, 0, v_, u_]
                        if np.abs(d_img - d_) < d_img * 0.1 + 0.5:  # error tolerance in meters
                            world_point_visibility[cam, i] = 1
                # fig = plt.figure(figsize=tuple(np.array(dataset.Rworld_shape)[::-1] / 50))
                # ax = fig.add_axes([0, 0, 1, 1])
                # ax.axis('off')
                # ax.imshow(world_point_visibility[cam].unflatten(0, list(
                #     map(lambda x: int(x / self.args.visibility_reduce), dataset.Rworld_shape))))
                # plt.savefig(f'cam{cam}_visibility.png')
                pass
            # rewards += world_point_visibility.mean(dim=1)
            mean_visibility = [world_point_visibility[:cam + 1].max(dim=0)[0].mean().item() for cam in range(N)]
            mean_visibility = to_tensor([0] + mean_visibility)
            rewards += mean_visibility[1:] - mean_visibility[:-1]

        return rewards, (overall_coverages, task_loss.item(), moda, min_dist)

    def get_advantages(self, b_step, b_configs, b_imgs, b_aug_mats, b_proj_mats,
                       b_actions, b_rewards, b_dones, b_imgs_inds):
        self.agent.eval()
        L, = b_step.shape
        B = self.args.rl_minibatch_size
        b_advantages = torch.zeros_like(b_rewards)
        b_returns = torch.zeros_like(b_rewards)
        b_values = []
        b_inds = torch.arange(L)
        with torch.no_grad():
            for start in range(0, L, B):  # drop_last=False
                end = min(start + B, L)
                mb_inds = b_inds[start:end]
                _, mb_values, _, _ = self.agent.get_action_and_value(
                    (b_step[mb_inds],
                     b_configs[mb_inds].cuda(),
                     b_imgs[b_imgs_inds[mb_inds]].cuda(),
                     b_aug_mats[mb_inds],
                     b_proj_mats[mb_inds]),
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
        return b_advantages, b_returns

    def train_rl(self, dataset, optimizer, epoch_):
        # flatten the batch
        b_step, b_configs, b_imgs, b_aug_mats, b_proj_mats = [], [], [], [], []
        for (step, configs, img, aug_mats, proj_mats) in self.memory_bank['obs']:
            if img is not None:
                b_imgs.append(img)  # ppo_steps / dataset.num_cam * (dataset.num_cam - 1)
            b_step.append(step)  # ppo_steps
            b_configs.append(configs)  # ppo_steps
            b_aug_mats.append(aug_mats)  # ppo_steps
            b_proj_mats.append(proj_mats)  # ppo_steps
        b_step, b_configs, b_imgs = torch.cat(b_step), torch.cat(b_configs), torch.cat(b_imgs)
        b_aug_mats, b_proj_mats = torch.cat(b_aug_mats), torch.cat(b_proj_mats)
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
                            }

        # only (L * (dataset.num_cam - 1) / dataset.num_cam) per-camera features are stored to save memory
        L, = b_step.shape
        _, N, _ = b_configs.shape
        _, _, H, W = b_imgs.shape
        B = self.args.rl_minibatch_size
        # append padding terms
        b_imgs = torch.cat([b_imgs, torch.zeros([1, 3, H, W])])
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
        b_imgs_inds = (idx_lookup[:, None].repeat(1, N) + idx_add_table[b_step]) * (
                idx_add_table[b_step] <= 0) - (idx_add_table[b_step] > 0)
        # idx for action_history
        idx_add_table = np.arange(N)[None, :] - np.arange(N)[:, None]
        b_action_history_inds = torch.arange(L)[:, None].repeat([1, N]) + idx_add_table[b_step]
        b_world_gt_idx = torch.tensor([(b_step[:i + 1] == 0).sum().item() - 1 for i in range(L)])
        clipfracs = []
        # Optimizing the policy and value network
        for rl_epoch in range(self.args.rl_update_epochs):
            # re-calculate advantages
            b_advantages, b_returns = self.get_advantages(b_step, b_configs, b_imgs, b_aug_mats, b_proj_mats,
                                                          b_actions, b_rewards, b_dones, b_imgs_inds)
            # train agent
            self.agent.train()
            b_inds = torch.randperm(L)
            for start in range(0, L - B + 1, B):  # drop_last=True
                end = start + B
                mb_inds = b_inds[start:end]
                action, newvalue, probs, x_feat = self.agent.get_action_and_value(
                    (b_step[mb_inds],
                     b_configs[mb_inds].cuda(),
                     b_imgs[b_imgs_inds[mb_inds]].cuda(),
                     b_aug_mats[mb_inds],
                     b_proj_mats[mb_inds]),
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
                                ).sum(-1).mean() if self.args.action_mapping == 'tanh' \
                    else probs.entropy().sum(-1).mean()

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
                # min_dist = torch.zeros([B]).cuda()
                # for b in range(B):
                #     step = b_step[mb_inds][b].item()
                #     if step > 0:
                #         min_dist[b] += torch.min(action_dist[b, :step])
                # steps_div = torch.clamp(min_dist, 0, self.args.div_clamp).mean()
                # make sure cameras won't look directly outside
                delta_dir = torch.clamp((xy + 0.1 * delta_xy).norm(dim=-1) - xy.norm(dim=-1), 0, None).mean() / 0.1
                # action diversity in terms of \mu
                mu_div_ = torch.zeros([N]).cuda()
                for step in range(1, N):  # skip the initial step as that one should be the same guess
                    idx = b_step[mb_inds] == step
                    if idx.sum().item() > 1:
                        mu_div_[step] = probs.loc[idx].std(dim=0).mean()
                mu_div = torch.clamp(mu_div_, 0, self.args.div_clamp).mean()
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

                # regularization for absolute values
                tanh_abs_mean = torch.abs(torch.tanh(probs.loc) if not self.agent.clip_action else probs.loc)
                tanh_abs_loss = torch.clamp(tanh_abs_mean - 0.99, 0).mean() * 100

                loss = (pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef) * self.args.use_ppo + \
                       (-steps_div * self.args.steps_div_coef - mu_div * self.args.mu_div_coef +
                        delta_dir * self.args.dir_coef + recons_loss * self.args.autoencoder_coef + tanh_abs_loss
                        ) * self.args.reg_decay_factor ** ((epoch_ - 1) // self.args.reg_decay_epochs)

                if torch.isnan(loss):
                    print('**************** nan loss ****************')
                    print(pg_loss, v_loss, entropy_loss, steps_div, mu_div, delta_dir, recons_loss)
                if torch.isinf(loss):
                    print('**************** inf loss ****************')
                    print(pg_loss, v_loss, entropy_loss, steps_div, mu_div, delta_dir, recons_loss)
                # fix the action_std for the initial epochs
                # optimizer.param_groups[-1]['lr'] = (epoch_ > self.args.std_wait_epochs) * \
                #                                    self.args.control_lr * self.args.std_lr_ratio
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                optimizer.step()
                grad_norm = 0
                for p in self.agent.parameters():
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** (1. / 2)

                # with torch.no_grad():
                #     for param in self.agent.parameters():
                #         param.clamp_(-1, 1)

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
            self.writer.add_scalar("losses/recons_loss", recons_loss.item(), self.rl_global_step)
            self.writer.add_scalar("losses/grad_norm", grad_norm, self.rl_global_step)

        with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
            action_means = self.agent.expand_mean_actions(dataset)
            # action_space = dataset.base.env.opts['env_action_space'].split('-')
            print(f'v loss: {v_loss.item():.3f}, p loss: {pg_loss.item():.3f}, ent: {entropy_loss.item():.3f}, '
                  f'avg return: {b_returns.mean().item():.3f}\n'
                  f'mu: {action_means.detach().cpu().numpy()}\n'
                  f'sigma: {probs.scale.detach().cpu().mean(dim=0).numpy()}'
                  )
            # if torch.where(b_step[mb_inds] == 0)[0].numel():
            #     idx = torch.where(b_step[mb_inds] == 0)[0][0].item()
            #     mu = probs.loc.detach().cpu().numpy()
            #     sigma = probs.scale.detach().cpu().numpy()
            #     # alpha = probs.concentration1.detach().cpu().numpy()
            #     # beta = probs.concentration0.detach().cpu().numpy()
            #     print(f'step 0: mu: \t{mu[idx]} \n        sigma: \t{sigma[idx]}'
            #           # f'step 0: \talpha: \t{alpha[idx]} \n        \tbeta: \t{beta[idx]}'
            #           )

        del b_step, b_configs, b_imgs, b_aug_mats, b_proj_mats, b_actions, b_logprobs, b_advantages, b_returns, b_values

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
            if self.args.interactive:
                # NOTE: Already considered the reID usage in searching camera configs.
                #       However, it seems impossible to add the tracking task itself as an reward function.
                #       Tracking must have multiple frames to compute the IoU between the predicted and ground truth.
                #       But in the RL setup, one episode is consist of N camera configs of one scene. The reward 
                #       is given at the end of each scene as the single frame detection performance.

                # NOTE: The `world_id` and `imgs_id` seems only useful in the training code
                #       below for training a discriminative feature for identifying different pedestrians.

                # NOTE: Let the expand_episode return the `world_id` and `imgs_id` in interactive mode for tuning
                #       the reID feature. Otherwise it will throw an error
                self.agent.eval()
                feat, (world_feat, world_heatmap, world_offset, world_id), (world_gt, imgs_gt), action_history = self.expand_episode(
                    dataloader.dataset, (step, configs, imgs, aug_mats, proj_mats),
                    True, visualize=self.rl_global_step % 500 == 0,
                    override_action=(torch.rand([N, len(dataloader.dataset.action_names)]) * 2 - 1
                                     if self.args.random_search else None))
            else:
                (world_heatmap, world_offset, world_id), (imgs_heatmap, imgs_offset, imgs_wh, imgs_id) = \
                    self.model(imgs.cuda(), aug_mats, proj_mats)
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].flatten(0, 1)

            # Compute the current step
            step = batch_idx + (epoch - 1) * len(dataloader)

            # MVDet loss - (w_loss = center_loss + offset_loss + id_loss)
            if self.args.use_mse:
                w_loss = F.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device))
                if self.writer is not None:
                    self.writer.add_scalar("mvdet/w_loss", w_loss.item(), step)
            else:
                loss_w_hm = focal_loss(world_heatmap, world_gt['heatmap'])
                loss_w_off = regL1loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
                loss_w_id = regCEloss(world_id, world_gt['reg_mask'], world_gt['idx'], world_gt['pid'])
                w_loss = loss_w_hm + loss_w_off + self.args.id_ratio * loss_w_id

                # Add loss information into tensorboard for debug
                if self.writer is not None:
                    self.writer.add_scalar("mvdet/loss_w_hm", loss_w_hm.item(), step)
                    self.writer.add_scalar("mvdet/loss_w_off", loss_w_off.item(), step)
                    self.writer.add_scalar("mvdet/loss_w_id", loss_w_id.item(), step)
            
            # When not training camera configs, compute per-view detection losses, including 
            # center_loss, offset_loss and wh_loss. When searching best camera configs, we don't
            # enable this loss because NO PEDESTRIAN is spawned in the world in this stage.
            if not self.args.interactive:
                if self.args.use_mse:
                    img_loss = F.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device))
                    if self.writer is not None:
                        self.writer.add_scalar("mvdet/img_loss", img_loss.item(), step)
                else:
                    loss_img_hm = focal_loss(imgs_heatmap, imgs_gt['heatmap'])
                    loss_img_off = regL1loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
                    loss_img_wh = regL1loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])
                    loss_img_id = regCEloss(imgs_id, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['pid'])
                    img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1 + self.args.id_ratio * loss_img_id

                    # Add loss information into tensorboard for debug
                    if self.writer is not None:
                        self.writer.add_scalar("mvdet/loss_img_hm", loss_img_hm.item(), step)
                        self.writer.add_scalar("mvdet/loss_img_off", loss_img_off.item(), step)
                        self.writer.add_scalar("mvdet/loss_img_wh", loss_img_wh.item(), step)
                        self.writer.add_scalar("mvdet/loss_img_id", loss_img_id.item(), step)
            else:
                img_loss = torch.zeros([]).cuda()

            loss = w_loss + img_loss / N * self.args.alpha
            losses += loss.item()

            if self.writer:
                self.writer.add_scalar("mvdet/loss", loss.item(), step)

            # train MVDet
            optimizers[0].zero_grad()
            loss.backward()
            optimizers[0].step()

            # train MVcontrol
            if len(self.memory_bank['obs']) >= self.args.ppo_steps:
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

    def test(self, dataloader, epoch=None, override_action=None, visualize=False):
        if override_action is not None:
            print(override_action)
        self.model.eval()
        t0 = time.time()
        losses = 0
        cover_avg = 0
        res_list = []
        track_res_list = []
        tracker_reset_frames = []

        for batch_idx, (step, configs, imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) in enumerate(dataloader):
            B, N = configs.shape[:2]
            # We need to assert batch_size == 1 for the tracking results gathering resons
            assert B == 1, 'only support batch_size == 1'
            # Initialize the test tracker for the tracking task
            if self.args.reID:
                # When the remainder is 0, it means that we have entered a new scene
                if batch_idx % dataloader.dataset.tracking_scene_len == 0:
                    # Reset the test_tracker each time we enter a new scene
                    self.test_tracker = JDETracker(conf_thres=self.args.tracker_conf_thres,
                                                   gating_threshold=self.args.tracker_gating_threshold)
                    # Remember the frames when the test_tracker is reset, used later for splitting the tracking results
                    tracker_reset_frames.append(frame.item())
 
            with torch.no_grad():
                if self.args.interactive:
                    self.agent.eval()
                    assert B == 1, 'only support batch_size/num_envs == 1'
                    feat, (world_feat, world_heatmap, world_offset, world_id), (world_gt, imgs_gt), action_history = self.expand_episode(
                        dataloader.dataset, (step, configs, imgs, aug_mats, proj_mats),
                        visualize=(batch_idx < 5), batch_idx=batch_idx, override_action=override_action)
                    # NOTE: Record the first action and use it for the all future steps for interactive mode
                    if override_action is None:
                        override_action = action_history
                else:
                    feat, _ = self.model.get_feat(imgs.cuda(), aug_mats, proj_mats)
                    world_feat, (world_heatmap, world_offset, world_id) = self.model.get_output(feat.cuda())
                    if batch_idx < 5 and visualize:
                        Image.fromarray(cover_visualize(dataloader.dataset, feat[0], world_heatmap[0],
                                                        {key: value[0] for key, value in world_gt.items()})
                                        ).save(f'{self.logdir}/cover_{batch_idx}.png')
                        save_image(make_grid(imgs[0], 6, normalize=True), f'{self.logdir}/imgs_{batch_idx}.png')
                # coverage
                cam_coverages = feat.norm(dim=2).bool().float()
                overall_coverages = cam_coverages.max(dim=1)[0].mean().item()
            cover_avg += overall_coverages
            loss = focal_loss(world_heatmap, world_gt['heatmap'])
            if self.args.use_mse:
                loss = F.mse_loss(world_heatmap, world_gt['heatmap'].cuda())
            losses += loss.item()

            # If reID mode, xys is [B, H*W, 3+128], otherwise [B, H*W, 3].
            # The first 3 channels are [x,y,s], last 128 channels are the embeddings.
            ids_emb = world_feat if self.args.reID else None
            xys = mvdet_decode(torch.sigmoid(world_heatmap), world_offset, ids_emb=ids_emb,
                               reduce=dataloader.dataset.world_reduce).cpu()

            # BEV detection predictions
            grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
            positions = grid_xy
            for b in range(B):
                ids = scores[b].squeeze() > self.args.cls_thres
                pos, s = positions[b, ids], scores[b, ids, 0]
                ids, count = nms(pos, s, 20, np.inf)
                res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                res_list.append(res)

            # Tracking predictions
            if self.args.reID:
                # bev_det is the [B, H*W, 5] tensor, where the last dimension is [x, y, _, _, score]
                bev_det = torch.concatenate([xys[:, :, :2], torch.ones_like(xys[:, :, :2]), xys[:, :, 2:3]], dim=2)
                # The id_emb is the [B, H*W, 128] tensor
                id_embs = xys[:, :, 3:].numpy()

                # iterating through batches
                for b in range(B):
                    # Although in the tracker, we will filter the ids based on the threshold, we do it here in advance.
                    ids = bev_det[b, :, 4] > self.args.tracker_conf_thres
                    # Pre-filtering out, each with shape [k, 5] and [k, 128], k is number of remaining indices
                    bev_det_subset = bev_det[b, ids]
                    id_embs_subset = id_embs[b, ids]
                    # Find the positions and scores and apply non-maximum-suppresion
                    pos, s = bev_det_subset[:, :2], bev_det_subset[:, 4]
                    # Applying NMS, and only use a subset of candidates
                    ids, count = nms(pos, s, 20, np.inf)
                    # We should filter the true_id by using the first k count in ids
                    true_ids = ids[:count]
                    # Put the bev_detection, id_embs into the tracker, which predicts the tracking results
                    output_stracks = self.test_tracker.update(bev_det_subset[true_ids], id_embs_subset[true_ids])
                    # The tracking results are stored in the track_res_list, as the score is not used, we don't save it.
                    track_res_list.extend([
                        torch.tensor([frame[b], s.track_id] + s.tlwh.tolist()[:2])
                        for s in output_stracks
                    ])

        # Always add the last (frame+1) as the reset frame, this ensures that last frame from the 
        # last sequence is included in the tracking results.
        tracker_reset_frames.append(frame.item() + 1)

        res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        # np.savetxt(f'{self.logdir}/test.txt', res, '%d')
        moda, modp, precision, recall, stats = evaluateDetection_py(res,
                                                                    dataloader.dataset.get_gt_array(),
                                                                    dataloader.dataset.frames)
        
        # Add MOTA during testing
        track_res = torch.stack(track_res_list, dim=0).numpy() if track_res_list else np.empty([0, 4])
        motas, motps, track_precisions, track_recalls = [], [], [], []

        # Evaluate tracking performance and convert to percentage
        # https://github.com/cheind/py-motmetrics/blob/develop/Readme.md
        # "Metric MOTP seems to be off. To convert compute (1 - MOTP) * 100.
        #   MOTChallenge benchmarks compute MOTP as percentage, while py-motmetrics sticks to
        #   the original definition of average distance over number of assigned objects [1]."

        # Split the testing according to each scene, since the MOTA/MOTP should be calculated for each scene
        # We split the prediction based on the scene length.
        for start, end in zip(tracker_reset_frames[:-1], tracker_reset_frames[1:]):
            # Frames that satisfies the condition
            t = track_res[(track_res[:, 0] >= start) & (track_res[:, 0] < end)]
            gt = dataloader.dataset.get_gt_array(reID=True)
            gt = gt[(gt[:, 0] >= start) & (gt[:, 0] < end)]

            # Compute the MOTA, MOTP, precision, recall
            summary = mot_metrics_pedestrian(t, gt)
            motas.append(summary['mota'].item() * 100)
            motps.append((1 - summary['motp'].item()) * 100)
            track_precisions.append(summary['precision'].item() * 100)
            track_recalls.append(summary['recall'].item() * 100)

        # Average the MOTA, MOTP, precision, recall
        mota = np.mean(motas) if motas else 0.
        motp = np.mean(motps) if motps else 0.
        track_precision = np.mean(track_precisions) if track_precisions else 0.
        track_recall = np.mean(track_recalls) if track_recalls else 0.

        print(f'Test, cover: {cover_avg / len(dataloader):.3f}, loss: {losses / len(dataloader):.6f}, '
              f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%, '
              f'mota: {mota:.1f}%, motp: {motp:.1f}%, prec_t: {track_precision:.1f}%, recall_t: {track_recall:.1f}%, '
              f'time: {time.time() - t0:.1f}s') 

        if self.writer is not None:
            self.writer.add_scalar("results/moda", moda, self.rl_global_step if self.args.interactive else epoch)
            self.writer.add_scalar("results/modp", modp, self.rl_global_step if self.args.interactive else epoch)
            self.writer.add_scalar("results/precision", precision, self.rl_global_step if self.args.interactive else epoch)
            self.writer.add_scalar("results/recall", recall, self.rl_global_step if self.args.interactive else epoch)
            if self.args.reID:
                self.writer.add_scalar("results/mota", mota, self.rl_global_step if self.args.interactive else epoch)
                self.writer.add_scalar("results/motp", motp, self.rl_global_step if self.args.interactive else epoch)
                self.writer.add_scalar("results/precision_t", track_precision, self.rl_global_step if self.args.interactive else epoch)
                self.writer.add_scalar("results/recall_t", track_recall, self.rl_global_step if self.args.interactive else epoch)

        return losses / len(dataloader), [moda, modp, precision, recall]#
