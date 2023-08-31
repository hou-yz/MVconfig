import os

os.environ['OMP_NUM_THREADS'] = '1'
import time
import itertools
import json
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.datasets import *
from src.models.mvdet import MVDet
from src.utils.logger import Logger
from src.utils.draw_curve import draw_curve
from src.utils.str2bool import str2bool
from src.trainer import PerspectiveTrainer


def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
        torch.autograd.set_detect_anomaly(True)
    else:
        print('No sys.gettrace')
        is_debug = False

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    if args.dataset == 'carlax':
        with open(f'./cfg/RL/{args.carla_cfg}.cfg', "r") as fp:
            dataset_config = json.load(fp)
        base = CarlaX(dataset_config, port=args.carla_port, tm_port=args.carla_tm_port)
        args.num_workers = 0
        args.batch_size = 1
    else:
        if args.dataset == 'wildtrack':
            base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
        elif args.dataset == 'multiviewx':
            base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
        else:
            raise Exception('must choose from [wildtrack, multiviewx]')
        args.batch_size = 1 if args.batch_size is None else args.batch_size
        args.interactive = False

    train_set = frameDataset(base, split='trainval', world_reduce=args.world_reduce,
                             img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                             img_kernel_size=args.img_kernel_size, interactive=args.interactive,
                             augmentation=args.augmentation and not args.interactive,
                             seed=None if args.interactive else args.carla_seed,  # random in interactive carla training
                             )
    test_set = frameDataset(base, split='test', world_reduce=args.world_reduce,
                            img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                            img_kernel_size=args.img_kernel_size,
                            interactive=args.interactive, seed=args.carla_seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    # logging
    RL_settings = f'RL_{args.carla_cfg}_{args.reward}_{"C" if args.control_arch == "conv" else "T"}_' \
                  f'steps{args.ppo_steps}_b{args.rl_minibatch_size}_e{args.rl_update_epochs}_lr{args.control_lr}_' \
                  f'stdinit{args.actstd_init}lr{args.actstd_lr}_ent{args.ent_coef}_' \
                  f'{"det_" if args.rl_deterministic else ""}' if args.interactive else ''
    logdir = f'logs/{args.dataset}/{"DEBUG_" if is_debug else ""}{RL_settings}' \
             f'TASK_{args.aggregation}_e{args.epochs}_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}' if not args.eval \
        else f'logs/{args.dataset}/EVAL_{args.resume}'
    os.makedirs(logdir, exist_ok=True)
    copy_tree('src', logdir + '/scripts/src')
    for script in os.listdir('.'):
        if script.split('.')[-1] == 'py':
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print(logdir)
    print('Settings:')
    print(vars(args))

    # model
    model = MVDet(train_set, args.arch, args.aggregation,
                  args.use_bottleneck, args.hidden_dim, args.outfeat_dim, args.control_arch, args.actstd_init).cuda()

    # load checkpoint
    if args.interactive:
        with open(f'logs/{args.dataset}/{args.arch}_.txt', 'r') as fp:
            load_dir = fp.read()
        print(load_dir)
        pretrained_dict = torch.load(f'{load_dir}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'control' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # tensorboard logging
        writer = SummaryWriter(logdir)
        writer.add_text("hyperparameters",
                        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|"
                                                                 for key, value in vars(args).items()])))
    else:
        if not args.eval:
            with open(f'logs/{args.dataset}/{args.arch}_.txt', 'w') as fp:
                fp.write(logdir)

        writer = None

    if args.resume:
        print(f'loading checkpoint: logs/{args.dataset}/{args.resume}')
        pretrained_dict = torch.load(f'logs/{args.dataset}/{args.resume}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    param_dicts = [{"params": [p for n, p in model.named_parameters()
                               if 'base' not in n and 'control' not in n and p.requires_grad],
                    "lr": args.lr * args.other_lr_ratio, } if not args.interactive else {"params": [], "lr": 0},
                   {"params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, } if not args.interactive else {"params": [], "lr": 0},
                   {"params": [p for n, p in model.named_parameters()
                               if 'control' in n and 'logstd' not in n and p.requires_grad],
                    "lr": args.control_lr, },
                   {"params": [p for n, p in model.named_parameters()
                               if 'control' in n and 'logstd' in n and p.requires_grad],
                    "lr": args.actstd_lr, }, ]
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    def warmup_lr_scheduler(epoch, warmup_epochs=0.1 * args.epochs):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler) if not args.interactive \
        else torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)

    trainer = PerspectiveTrainer(model, logdir, writer, args)

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []

    # learn
    if not args.eval:
        if not is_debug:
            trainer.test(test_loader)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, scheduler)
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()
            print('Testing...')
            test_loss, test_prec = trainer.test(test_loader)

            # draw & save
            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec[0])
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s,
                       train_prec_s, test_prec_s)
            torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))

    print('Test loaded model...')
    print(logdir)
    trainer.test(test_loader)
    if args.interactive:
        base.env.close()
        writer.close()


if __name__ == '__main__':
    # common settings
    parser = argparse.ArgumentParser(description='view control for multiview detection')
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--aggregation', type=str, default='max', choices=['mean', 'max'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack',
                        choices=['wildtrack', 'multiviewx', 'carlax'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='input batch size for training')
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for task network')
    parser.add_argument('--base_lr_ratio', type=float, default=1.0)
    parser.add_argument('--other_lr_ratio', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--carla_seed', type=int, default=2023, help='random seed for CarlaX')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    # MVcontrol settings
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--carla_cfg', type=str, default='1')
    parser.add_argument('--control_arch', default='conv', choices=['conv', 'transformer'])
    parser.add_argument('--rl_deterministic', type=str2bool, default=False)
    parser.add_argument('--carla_port', type=int, default=2000)
    parser.add_argument('--carla_tm_port', type=int, default=8000)
    # RL arguments
    parser.add_argument('--control_lr', type=float, default=3e-4, help='learning rate for MVcontrol')
    parser.add_argument('--actstd_lr', type=float, default=3e-3, help='learning rate for actor std')
    # https://arxiv.org/abs/2006.05990
    parser.add_argument('--actstd_init', type=float, default=0.5, help='initial value actor std')
    parser.add_argument("--reward", default='moda')
    # https://www.reddit.com/r/reinforcementlearning/comments/n09ns2/explain_why_ppo_fails_at_this_very_simple_task/
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    parser.add_argument("--ppo_steps", type=int, default=256,
                        help="the number of steps to run in each environment per policy rollout, default: 2048")
    parser.add_argument("--rl_minibatch_size", type=int, default=32,
                        help="RL mini-batches, default: 64")
    parser.add_argument("--rl_update_epochs", type=int, default=5,
                        help="the K epochs to update the policy, default: 10")
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor, default: 0.99')
    parser.add_argument("--gae", type=str2bool, default=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--norm_adv", type=str2bool, default=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=str2bool, default=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent_coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="the target KL divergence threshold")
    # multiview detection specific settings
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--augmentation', type=str2bool, default=True)
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)

    args = parser.parse_args()

    main(args)
