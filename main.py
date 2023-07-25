import os

os.environ['OMP_NUM_THREADS'] = '1'
import time
import itertools
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
from src.datasets import *
from src.models.mvdet import MVDet
from src.models.mvcnn import MVCNN
from src.utils.logger import Logger
from src.utils.draw_curve import draw_curve
from src.utils.str2bool import str2bool
from src.trainer import PerspectiveTrainer, find_dataset_lvl_strategy


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

        import json

        with open('./cfg/RL/1.cfg', "r") as fp:
            dataset_config = json.load(fp)
        base = CarlaX(dataset_config, args.carla_seed)

        args.task = 'mvdet'
        args.num_workers = 0
        result_type = ['moda', 'modp', 'prec', 'recall']
        args.lr = 5e-4 if args.lr is None else args.lr
        args.select_lr = 1e-4 if args.select_lr is None else args.select_lr
        args.batch_size = 1 if args.batch_size is None else args.batch_size

        train_set = frameDataset(base, split='trainval', world_reduce=args.world_reduce,
                                 img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                                 img_kernel_size=args.img_kernel_size,
                                 dropout=args.dropcam, augmentation=args.augmentation)
        val_set = frameDataset(base, split='val', world_reduce=args.world_reduce,
                               img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                               img_kernel_size=args.img_kernel_size)
        test_set = frameDataset(base, split='test', world_reduce=args.world_reduce,
                                img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                                img_kernel_size=args.img_kernel_size)
    else:
        if args.dataset == 'wildtrack':
            base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
        elif args.dataset == 'multiviewx':
            base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
        else:
            raise Exception('must choose from [wildtrack, multiviewx]')

        args.task = 'mvdet'
        result_type = ['moda', 'modp', 'prec', 'recall']
        args.lr = 5e-4 if args.lr is None else args.lr
        args.select_lr = 1e-4 if args.select_lr is None else args.select_lr
        args.batch_size = 1 if args.batch_size is None else args.batch_size

        train_set = frameDataset(base, split='trainval', world_reduce=args.world_reduce,
                                 img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                                 img_kernel_size=args.img_kernel_size,
                                 dropout=args.dropcam, augmentation=args.augmentation)
        val_set = frameDataset(base, split='val', world_reduce=args.world_reduce,
                               img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                               img_kernel_size=args.img_kernel_size)
        test_set = frameDataset(base, split='test', world_reduce=args.world_reduce,
                                img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                                img_kernel_size=args.img_kernel_size)

    if args.interactive:
        args.lr /= 5
        # args.epochs *= 2

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)
    N = train_set.num_cam

    # logging
    lr_settings = f'base{args.base_lr_ratio}other{args.other_lr_ratio}' + \
                  f'select{args.select_lr}' if args.interactive else ''
    logdir = f'logs/{args.dataset}/{"DEBUG_" if is_debug else ""}{args.arch}_{args.aggregation}_down{args.down}_' \
             f'{"RL_" if args.interactive else ""}' \
             f'lr{args.lr}{lr_settings}_b{args.batch_size}_e{args.epochs}_dropcam{args.dropcam}_' \
             f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}' if not args.eval \
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
                  args.use_bottleneck, args.hidden_dim, args.outfeat_dim).cuda()

    # load checkpoint
    if args.interactive:
        with open(f'logs/{args.dataset}/{args.arch}_performance.txt', 'r') as fp:
            result_str = fp.read()
        print(result_str)
        load_dir = result_str.split('\n')[1].replace('# ', '')
        pretrained_dict = torch.load(f'{load_dir}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'select' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.resume:
        print(f'loading checkpoint: logs/{args.dataset}/{args.resume}')
        pretrained_dict = torch.load(f'logs/{args.dataset}/{args.resume}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    param_dicts = [{"params": [p for n, p in model.named_parameters()
                               if 'base' not in n and 'select' not in n and p.requires_grad],
                    "lr": args.lr * args.other_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters() if 'select' in n and p.requires_grad],
                    "lr": args.select_lr, }, ]
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    def warmup_lr_scheduler(epoch, warmup_epochs=0.1 * args.epochs):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)

    trainer = PerspectiveTrainer(model, logdir, args)

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []

    # learn
    if not args.eval:
        # trainer.test(test_loader)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, scheduler)
            if epoch % max(args.epochs // 10, 1) == 0:
                print('Testing...')
                test_loss, test_prec = trainer.test(test_loader, torch.eye(N) if args.interactive else None)

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
    # if args.eval:
    #     trainer.test(test_loader, torch.eye(N))
    trainer.test(test_loader)


if __name__ == '__main__':
    # common settings
    parser = argparse.ArgumentParser(description='view selection for multiview classification & detection')
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--aggregation', type=str, default='max', choices=['mean', 'max'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack',
                        choices=['wildtrack', 'multiviewx', 'modelnet40_12', 'modelnet40_20', 'scanobjectnn', 'carlax'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='input batch size for training')
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='learning rate for task network')
    parser.add_argument('--select_lr', type=float, default=None, help='learning rate for MVselect')
    parser.add_argument('--base_lr_ratio', type=float, default=1.0)
    parser.add_argument('--other_lr_ratio', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--carla_seed', type=int, default=2023, help='random seed for CarlaX')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    # MVcontrol settings
    parser.add_argument('--interactive', type=str2bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor (default: 0.99)')
    parser.add_argument('--down', type=int, default=1, help='down sample the image to 1/N size')
    # parser.add_argument('--beta_entropy', type=float, default=0.01)
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
