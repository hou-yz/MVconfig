import io
import itertools
import random
import time
import copy
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from PIL import Image
from src.loss import *
from src.evaluation.evaluate import evaluate, evaluateDetection_py
from src.models.mvcontrol import expectation, tanh_prime
from src.utils.decode import ctdet_decode, mvdet_decode
from src.utils.nms import nms
from src.utils.meters import AverageMeter
from src.utils.image_utils import add_heatmap_to_image, img_color_denormalize


class PerspectiveTrainer(object):
    def __init__(self, model, logdir, writer, args, ):
        self.model = model
        self.agent = model.control_module
        self.args = args
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.rl_global_step = 0
        self.writer = writer
        self.start_time = time.time()
        self.memory_bank = {'obs': [],
                            'actions': [],
                            'logprobs': [],
                            # 'rewards': [],
                            'dones': [],
                            'values': [],
                            'returns': [],
                            'advantages': []}
        # beta distribution in agent will output [0, 1], whereas the env has action space of [-1, 1]
        # self.action_mapping = lambda x: torch.clamp(x, -1, 1)
        self.action_mapping = torch.tanh

    # https://github.com/vwxyzjn/ppo-implementation-details
    def expand_episode(self, env, init_obs, training=False, visualize=False):
        step, configs, _, _, _ = init_obs
        B, = step.shape
        assert B == 1, 'currently only support batch size of 1 for the envs'
        next_done = False
        obs_feat = torch.zeros([B, env.num_cam, self.model.base_dim, *env.Rworld_shape]).cuda()
        imgs = []
        action_history = []
        # step 0: initialization
        cam_feat = torch.zeros([B, 1, self.model.base_dim, *env.Rworld_shape]).cuda()
        while not next_done:
            # step 1 ~ N: action
            with torch.no_grad():
                action, value, probs = self.agent.get_action_and_value(
                    (obs_feat, configs, step), deterministic=self.args.rl_deterministic and not training)
            if training:
                self.rl_global_step += 1 * B
                # Markovian if have (obs_feat, configs, step) as state
                # only store cam_feat (one cam) instead of obs_feat (all cams) to save memory
                self.memory_bank['obs'].append((cam_feat[:, 0].cpu() if step.item() != 0 else None, configs, step))
                self.memory_bank['dones'].append(next_done)
                self.memory_bank['actions'].append(action.cpu())
                self.memory_bank['values'].append(value.item())
                self.memory_bank['logprobs'].append(probs.log_prob(action).sum(-1).item())

            (step, configs, img, aug_mat, proj_mat, world_gt, img_gt, frame), next_done = \
                env.step(self.action_mapping(action[0]).cpu().numpy())
            imgs.append(img)
            action_history.append(action.cpu())
            step, configs, img, aug_mat, proj_mat = (torch.tensor(step)[None],
                                                     torch.tensor(configs, dtype=torch.float32)[None, :],
                                                     img[None, :], aug_mat[None, :], proj_mat[None, :])
            with torch.no_grad():
                cam_feat, _ = self.model.get_feat(img.cuda(), aug_mat, proj_mat)
            cam = step - 1
            obs_feat[0, cam] = cam_feat[0]

        # visualize
        def cover_visualize():
            cover_map = obs_feat[0].norm(dim=1).bool().float().mean([0]).cpu()
            pedestrian_gt_ij = torch.where(world_gt['heatmap'][0] == 1)
            fig = plt.figure(figsize=tuple(np.array(env.Rworld_shape)[::-1] / 50))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ax.imshow(cover_map + torch.sigmoid(world_heatmap)[0, 0].detach().cpu(), vmin=0, vmax=2)
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

        if not training:
            world_heatmap, world_offset = self.model.get_output(obs_feat)
            if visualize:
                Image.fromarray(cover_visualize()).save(f'{self.logdir}/cover.png')
                save_image(make_grid(torch.cat(imgs), normalize=True), f'{self.logdir}/imgs.png')
            return obs_feat.cpu(), (world_heatmap, world_offset)

        rewards, stats, world_heatmap = self.rl_rewards(env, action_history, obs_feat[0].cpu(), world_gt, frame)
        coverages, task_loss, moda, min_dist = stats
        # fixed episode length of num_cam - 1 so no need for value bootstrap
        returns, advantages = np.zeros([env.num_cam]), np.zeros([env.num_cam])
        values = np.array(self.memory_bank['values'][-env.num_cam:])
        lastgaelam = 0
        for t in range(-1, -(env.num_cam + 1), -1):
            if t == -1:  # last item [-1]
                nextnonterminal = 1.0 - next_done
                next_value = next_return = 0
            else:  # second and third last item [-2], [-3], ...
                nextnonterminal = 1.0 - self.memory_bank['dones'][t + 1]
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
        for t in range(env.num_cam):
            # self.memory_bank['rewards'].append(float(rewards[t]))
            self.memory_bank['advantages'].append(float(advantages[t]))
            self.memory_bank['returns'].append(float(returns[t]))

        self.writer.add_scalar("charts/episodic_return", rewards.sum().item(), self.rl_global_step)
        self.writer.add_scalar("charts/episodic_length", step.item(), self.rl_global_step)
        self.writer.add_scalar("charts/coverage", coverages[-1].item(), self.rl_global_step)
        self.writer.add_scalar("charts/action_dist", min_dist.mean().item(), self.rl_global_step)
        self.writer.add_scalar("charts/loss", task_loss, self.rl_global_step)
        self.writer.add_scalar("charts/moda", moda, self.rl_global_step)
        if visualize:
            self.writer.add_image("images/coverage", cover_visualize(), self.rl_global_step, dataformats='HWC')
            # self.writer.add_image("images/imgs", make_grid(torch.cat(imgs), normalize=True),
            #                       self.rl_global_step, dataformats='CHW')

        return

    # https://github.com/vwxyzjn/ppo-implementation-details
    def rl_rewards(self, env, action_history, obs_feat, world_gt, frame):
        # coverage
        cam_coverages = torch.cat([torch.zeros([1, *env.Rworld_shape]), obs_feat.norm(dim=1).bool().float()])
        overall_coverages = torch.stack([cam_coverages[:cam + 1].max(dim=0)[0].mean()
                                         for cam in range(env.num_cam + 1)])
        # compute loss & moda based on final result
        task_losses = torch.zeros([env.num_cam + 1])
        modas = torch.zeros([env.num_cam + 1])
        for cam in range(env.num_cam):
            world_heatmap, world_offset = self.model.get_output(obs_feat[None, :cam + 1].cuda())
            # loss
            task_losses[cam + 1] = focal_loss(world_heatmap, world_gt['heatmap'][None, :])
            # MODA
            xys = mvdet_decode(torch.sigmoid(world_heatmap), world_offset, reduce=env.world_reduce).cpu()
            grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
            if env.base.indexing == 'xy':
                positions = grid_xy
            else:
                positions = grid_xy[:, :, [1, 0]]
            ids = scores[0].squeeze() > self.args.cls_thres
            pos, s = positions[0, ids], scores[0, ids, 0]
            ids, count = nms(pos, s, 20, np.inf)
            res = torch.cat([torch.ones([count, 1]) * frame, pos[ids[:count]]], dim=1)
            moda, modp, precision, recall, stats = evaluateDetection_py(res, env.get_gt_array([frame]), [frame])
            modas[cam + 1] = moda
        # diversity
        action_history = torch.cat(action_history)
        action_dist = ((self.action_mapping(action_history[:, None]) -
                        self.action_mapping(action_history[None])) ** 2).sum(-1) ** 0.5
        min_dist = torch.zeros([env.num_cam])
        for i in range(1, env.num_cam):
            min_dist[i] += torch.min(action_dist[i, :i])
        # use coverage, loss, or MODA as reward
        rewards = torch.zeros([env.num_cam])
        if 'maxcover' in self.args.reward:
            # rewards += (overall_coverages[1:] - overall_coverages[:-1]) * 0.1
            rewards[-1] += overall_coverages[-1] * 0.1  # final step
        if 'avgcover' in self.args.reward:  # dense
            rewards += cam_coverages.mean(dim=[1, 2])[1:] * 0.1  # dense
        if 'loss' in self.args.reward:
            rewards += (-task_losses[1:] + task_losses[:-1])  # dense
        if 'moda' in self.args.reward:
            # rewards += (modas[1:] - modas[:-1]) / 100
            rewards[-1] += modas[-1] / 100  # final step
        # encourage each action to be more dis-similar
        if 'div' in self.args.reward:
            rewards += min_dist * 0.01

        return rewards, (overall_coverages, task_losses[-1].item(), modas[-1].item(), min_dist), world_heatmap

    def train_rl(self, env, optimizer):
        # flatten the batch
        b_cam_feat, b_configs, b_step = [], [], []
        for (cam_feat, configs, step) in self.memory_bank['obs']:
            if cam_feat is not None:
                b_cam_feat.append(cam_feat)  # ppo_steps / env.num_cam * (env.num_cam - 1)
            b_configs.append(configs)  # ppo_steps
            b_step.append(step)  # ppo_steps
        b_cam_feat, b_configs, b_step = torch.cat(b_cam_feat), torch.cat(b_configs), torch.cat(b_step)
        b_actions = torch.cat(self.memory_bank['actions'])
        b_logprobs = torch.tensor(self.memory_bank['logprobs'])
        b_advantages = torch.tensor(self.memory_bank['advantages'])
        b_returns = torch.tensor(self.memory_bank['returns'])
        b_values = torch.tensor(self.memory_bank['values'])

        # reset memory bank
        self.memory_bank = {'obs': [],
                            'actions': [],
                            'logprobs': [],
                            # 'rewards': [],
                            'dones': [],
                            'values': [],
                            'returns': [],
                            'advantages': []}

        # only (N * (env.num_cam - 1) / env.num_cam) per-camera features are stored to save memory
        N, = b_step.shape
        _, C, H, W = b_cam_feat.shape
        b_cam_feat = torch.cat([b_cam_feat, torch.zeros([1, C, H, W])])  # set -1 as the all zero term
        # idx for cam_feat
        idx_lookup = torch.stack([(b_step > 0)[:n + 1].sum() for n in range(N)]) - 1
        # where to find cam_feat indices in obs_feat
        # if positive, then it should be all zeros for that cam_feat
        # if zero, it should be the same as idx_lookup
        # if negative, find previous locations
        idx_add_table = np.arange(env.num_cam)[None, :] - np.arange(env.num_cam)[:, None]
        idx_add_table = np.concatenate([np.ones([1, env.num_cam], dtype=np.int64), idx_add_table])
        # if idx_add_table[b_step] <= 0, calculate the index for the non-zero cam_feats
        # else, the index should be -1
        b_feat_inds = (idx_lookup[:, None].repeat(1, env.num_cam) + idx_add_table[b_step]) * (
                idx_add_table[b_step] <= 0) - (idx_add_table[b_step] > 0)
        # idx for action_history
        idx_add_table = np.arange(env.num_cam)[None, :] - np.arange(env.num_cam)[:, None]
        b_action_history_inds = torch.arange(N)[:, None].repeat([1, env.num_cam]) + idx_add_table[b_step]
        clipfracs = []
        # Optimizing the policy and value network
        for epoch in range(self.args.rl_update_epochs):
            b_inds = torch.randperm(N)
            for start in range(0, N - self.args.rl_minibatch_size + 1, self.args.rl_minibatch_size):  # drop_last=True
                end = start + self.args.rl_minibatch_size
                mb_inds = b_inds[start:end]
                action, newvalue, probs = self.agent.get_action_and_value((b_cam_feat[b_feat_inds[mb_inds]].cuda(),
                                                                           b_configs[mb_inds],
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
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # https://arxiv.org/pdf/2006.05990.pdf
                # section B.8
                entropy_loss = (probs.entropy() +
                                expectation(probs, [probs.loc - 3 * probs.scale, probs.loc + 3 * probs.scale],
                                            tanh_prime, device='cuda')
                                ).sum(-1).mean()

                # div loss
                mb_action_history = b_actions[b_action_history_inds[mb_inds]].cuda()
                action_dist = ((self.action_mapping(probs.loc[:, None]) -
                                self.action_mapping(mb_action_history)) ** 2).sum(-1) ** 0.5
                min_dist = torch.zeros([self.args.rl_minibatch_size]).cuda()
                for i in range(self.args.rl_minibatch_size):
                    step = b_step[mb_inds][i].item()
                    if step > 0:
                        min_dist[i] += torch.min(action_dist[i, :step])
                div_loss = torch.clamp(min_dist, 0, self.args.div_clamp).mean()

                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef - \
                       div_loss * self.args.div_coef

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
        self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[-1]["lr"], self.rl_global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.rl_global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.rl_global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.rl_global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.rl_global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.rl_global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.rl_global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.rl_global_step)
        self.writer.add_scalar("losses/action_dist", min_dist.mean().item(), self.rl_global_step)
        self.writer.add_scalar("losses/div_loss", div_loss.item(), self.rl_global_step)

        with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
            # action_space = env.base.env.opts['env_action_space'].split('-')
            print(f'v loss: {v_loss.item():.3f}, p loss: {pg_loss.item():.3f}, ent: {entropy_loss.item():.3f}, '
                  f'action dist: {min_dist.mean().item():.3f}, div loss: {div_loss.item():.3f}, '
                  f'avg return: {b_returns.mean().item():.3f}')
            if torch.where(b_step[mb_inds] == 0)[0].numel():
                idx = torch.where(b_step[mb_inds] == 0)[0][0].item()
                mu = probs.loc.detach().cpu().numpy()
                sigma = probs.scale.detach().cpu().numpy()
                # alpha = probs.concentration1.detach().cpu().numpy()
                # beta = probs.concentration0.detach().cpu().numpy()
                print(f'step 0: mu: \t{mu[idx]} \n        sigma: \t{sigma[idx]}'
                      # f'step 0: \talpha: \t{alpha[idx]} \n        \tbeta: \t{beta[idx]}'
                      )

        del b_cam_feat, b_configs, b_step, b_actions, b_logprobs, b_advantages, b_returns, b_values

    def train(self, epoch, dataloader, optimizer, scheduler=None, log_interval=100):
        self.model.train()
        if self.args.base_lr_ratio == 0:
            self.model.base.eval()
        losses = 0
        t0 = time.time()
        for batch_idx, (step, configs, imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) in enumerate(dataloader):
            B, N = configs.shape[:2]
            if self.args.interactive:
                self.expand_episode(dataloader.dataset, (step, configs, imgs, aug_mats, proj_mats),
                                    True, visualize=self.rl_global_step % 500 == 0)
                if len(self.memory_bank['values']) >= self.args.ppo_steps:  # or batch_idx + 1 == len(dataloader)
                    self.train_rl(dataloader.dataset, optimizer)
            else:
                for key in imgs_gt.keys():
                    imgs_gt[key] = imgs_gt[key].flatten(0, 1)
                (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = \
                    self.model(imgs.cuda(), aug_mats, proj_mats)
                loss_w_hm = focal_loss(world_heatmap, world_gt['heatmap'])
                loss_w_off = regL1loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
                # loss_w_id = self.ce_loss(world_id, world_gt['reg_mask'], world_gt['idx'], world_gt['pid'])
                loss_img_hm = focal_loss(imgs_heatmap, imgs_gt['heatmap'])
                loss_img_off = regL1loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
                loss_img_wh = regL1loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])
                # loss_img_id = self.ce_loss(imgs_id, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['pid'])

                w_loss = loss_w_hm + loss_w_off  # + self.args.id_ratio * loss_w_id
                img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1  # + self.args.id_ratio * loss_img_id
                loss = w_loss + img_loss / N * self.args.alpha
                if self.args.use_mse:
                    loss = F.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device)) + \
                           self.args.alpha * F.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device)) / N

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

            if scheduler is not None:
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
        return losses / len(dataloader), None

    def test(self, dataloader):
        t0 = time.time()
        self.model.eval()
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
                        visualize=(batch_idx == 0))
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
            if dataloader.dataset.base.indexing == 'xy':
                positions = grid_xy
            else:
                positions = grid_xy[:, :, [1, 0]]
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
