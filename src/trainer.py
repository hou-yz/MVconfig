import itertools
import random
import time
import copy
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from src.loss import *
from src.evaluation.evaluate import evaluate, evaluateDetection_py
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
                            'obs_feat': [],
                            'actions': [],
                            'logprobs': [],
                            'rewards': [],
                            'dones': [],
                            'values': [],
                            'returns': [],
                            'advantages': []}

    def expand_episode(self, env, next_obs, training=False):
        B, = next_obs[0].shape
        assert B == 1, 'currently only support batch size of 1 for the envs'
        next_done = False
        obs_feat = torch.zeros([env.num_cam, self.model.base_dim, *self.model.Rworld_shape])
        while not next_done:
            step, configs, imgs, aug_mats, proj_mats = next_obs
            self.rl_global_step += 1 * B

            # ALGO LOGIC: action logic
            with torch.no_grad():
                feat, _ = self.model.get_feat(imgs.cuda(), aug_mats, proj_mats)
                obs_feat[step] += feat[0, 0].cpu()
                action, logprob, _, value = self.agent.get_action_and_value((obs_feat[None, :].cuda(), configs, step))
            if training:
                self.memory_bank['obs'].append(next_obs)
                self.memory_bank['dones'].append(next_done)
                self.memory_bank['obs_feat'].append(obs_feat)
                self.memory_bank['actions'].append(action)
                self.memory_bank['values'].append(value.item())
                self.memory_bank['logprobs'].append(logprob.item())

            # next_obs, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
            (step, configs, imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame), next_done = \
                env.step(action.cpu()[0].numpy())
            next_obs = (torch.tensor(step)[None], torch.tensor(configs, dtype=torch.float32)[None, :],
                        imgs[None, :], aug_mats[None, :], proj_mats[None, :])

        # last step & final update to obs_feat
        step, configs, imgs, aug_mats, proj_mats = next_obs
        with torch.no_grad():
            feat, _ = self.model.get_feat(imgs.cuda(), aug_mats, proj_mats)
            obs_feat[step] += feat[0, 0].cpu()

        if not training:
            return obs_feat

        rewards, (coverages, task_loss, moda) = self.rl_rewards(env, obs_feat, world_gt, frame)
        # fixed episode length of num_cam - 1 so no need for value bootstrap
        returns, advantages = np.zeros([env.num_cam - 1]), np.zeros([env.num_cam - 1])
        values = np.array(self.memory_bank['values'][-env.num_cam - 1:])
        lastgaelam = 0
        for t in range(1, env.num_cam):
            if t == 1:  # last item [-1]
                nextnonterminal = 1.0 - next_done
                next_value = next_return = 0
            else:  # second and third last item [-2], [-3], ...
                nextnonterminal = 1.0 - self.memory_bank['dones'][-(t - 1)]
                next_value = values[-(t - 1)]
                next_return = returns[-(t - 1)]
            if self.args.gae:
                delta = rewards[-t] + self.args.gamma * next_value * nextnonterminal - values[-t]
                advantages[-t] = lastgaelam = \
                    delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns[-t] = advantages[-t] + values[-t]
            else:
                returns[-t] = rewards[-t] + self.args.gamma * nextnonterminal * next_return
                advantages[-t] = returns[-t] - values[-t]
        for t in range(env.num_cam - 1):
            self.memory_bank['rewards'].append(rewards[t])
            self.memory_bank['advantages'].append(advantages[t])
            self.memory_bank['returns'].append(returns[t])

        self.writer.add_scalar("charts/episodic_return", rewards.sum().item(), self.rl_global_step)
        self.writer.add_scalar("charts/episodic_length", step.item(), self.rl_global_step)

        return coverages, task_loss, moda

    def rl_rewards(self, env, obs_feat, world_gt, frame):
        # coverage
        coverages = obs_feat.norm(dim=1).bool().float().mean(dim=[1, 2])
        # compute loss & moda based on final result
        # loss
        world_heatmap, world_offset = self.model.get_output(obs_feat[None, :].cuda())
        task_loss = focal_loss(world_heatmap, world_gt['heatmap'][None, :])
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
        moda, modp, precision, recall, stats = evaluateDetection_py(res, env.gt_array, env.frames)
        # use coverage, loss, or MODA as reward
        if self.args.reward == 'cover':
            rewards = coverages[-(env.num_cam - 1):]
        elif self.args.reward == 'loss':
            rewards = torch.zeros([env.num_cam - 1, ])
            rewards[-1] = task_loss
        elif self.args.reward == 'moda':
            rewards = torch.zeros([env.num_cam - 1, ])
            rewards[-1] = moda / 100
        else:
            raise Exception

        return rewards, (coverages, task_loss, moda)

    def train_rl(self, optimizer):
        # flatten the batch
        b_obs = tuple(torch.cat(obs) for obs in tuple(zip(*self.memory_bank['obs'])))
        b_step, b_configs, b_imgs, b_aug_mats, b_proj_mats = b_obs
        b_obs_feat = torch.stack(self.memory_bank['obs_feat'])
        b_actions = torch.cat(self.memory_bank['actions'])
        b_logprobs = torch.tensor(self.memory_bank['logprobs'])
        b_advantages = torch.tensor(self.memory_bank['advantages'])
        b_returns = torch.tensor(self.memory_bank['returns'])
        b_values = torch.tensor(self.memory_bank['values'])

        # Optimizing the policy and value network
        b_inds = np.arange(len(b_values))
        clipfracs = []
        for epoch in range(self.args.rl_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.rl_batch_size, self.args.rl_minibatch_size):
                end = start + self.args.rl_minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = \
                    self.agent.get_action_and_value((b_obs_feat[mb_inds].cuda(),
                                                     b_configs[mb_inds],
                                                     b_step[mb_inds]),
                                                    b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds].cuda()
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.rl_clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds].cuda()
                if self.args.rl_norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio,
                                                        1 - self.args.rl_clip_coef,
                                                        1 + self.args.rl_clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.rl_clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds].cuda()) ** 2
                    v_clipped = b_values[mb_inds].cuda() + torch.clamp(newvalue - b_values[mb_inds].cuda(),
                                                                       -self.args.rl_clip_coef,
                                                                       self.args.rl_clip_coef, )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds].cuda()) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.rl_ent_coef * entropy_loss + v_loss * self.args.rl_vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.rl_max_grad_norm)
                optimizer.step()

            if self.args.rl_target_kl is not None:
                if approx_kl > self.args.rl_target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # reset memory bank
        self.memory_bank = {'obs': [],
                            'obs_feat': [],
                            'actions': [],
                            'logprobs': [],
                            'rewards': [],
                            'dones': [],
                            'values': [],
                            'returns': [],
                            'advantages': []}

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[-1]["lr"], self.rl_global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.rl_global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.rl_global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.rl_global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.rl_global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.rl_global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.rl_global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.rl_global_step)
        self.writer.add_scalar("charts/SPS", int(self.rl_global_step / (time.time() - self.start_time)),
                               self.rl_global_step)
        print(f'v loss: {v_loss.item():.3f}, p loss: {pg_loss.item():.3f}, avg return: {b_returns.mean().item():.3f}')

    def train(self, epoch, dataloader, optimizer, scheduler=None, log_interval=100):
        self.model.train()
        if self.args.base_lr_ratio == 0:
            self.model.base.eval()
        losses = 0
        t0 = time.time()
        for batch_idx, (step, configs, imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].flatten(0, 1)
            if self.args.interactive:
                self.expand_episode(dataloader.dataset, (step, configs, imgs, aug_mats, proj_mats), True)
                if len(self.memory_bank['obs']) >= self.args.rl_batch_size:
                    self.train_rl(optimizer)
            else:
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
            B, N = imgs_gt['heatmap'].shape[:2]
            with torch.no_grad():
                if self.args.interactive:
                    feat = self.expand_episode(dataloader.dataset, (step, configs, imgs, aug_mats, proj_mats))
                else:
                    feat, _ = self.model.get_feat(imgs.cuda(), aug_mats, proj_mats)
                coverages = feat.norm(dim=1).bool().float().mean(dim=[1, 2])
                world_heatmap, world_offset = self.model.get_output(feat[None, :].cuda())
            cover_avg += coverages.mean().item()
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
                                                                    dataloader.dataset.gt_array,
                                                                    dataloader.dataset.frames)
        print(f'Test, cover: {cover_avg / len(dataloader):.3f}, loss: {losses / len(dataloader):.6f}, '
              f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%, '
              f'time: {time.time() - t0:.1f}s')

        return losses / len(dataloader), [moda, modp, precision, recall]
