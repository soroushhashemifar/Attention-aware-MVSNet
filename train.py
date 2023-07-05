import os
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from model import AttMVSNet
from utils import tocuda
import gc
import sys
import datetime
import torch.utils
import torch.utils.checkpoint
from torchscan import summary
from dataset import AttMVSDataset


class Args:
    mode = 'train'
    dataset_root = '/content/dtu-train-128/'
    imgsize = 128
    nsrc = 2
    nscale = 0
    ndepths = 256
    depth_min = 425
    depth_max = 935

    epochs = 50
    lr = 0.0001 # 0.001
    lrepochs = "10,20,30,40:2"
    wd = 0.0
    batch_size = 5
    summary_freq = 1
    save_freq = 1

    init_epoch = 18 # 0
    loadckpt = '/content/ATT_MVSNET/checkpoints/model_000017.ckpt'
    logckptdir = './checkpoints/'

    cuda = True


args = Args()

train_dataset = AttMVSDataset(args)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, drop_last=True)

model = AttMVSNet(args)
print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

if args.loadckpt:
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])

if args.cuda:
    model.cuda()

model.train()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=-1)

    last_loss = None
    this_loss = None
    for epoch_idx in range(args.init_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(train_loader) * epoch_idx

        if last_loss is None:
            last_loss = 999999
        else:
            last_loss = this_loss
        this_loss = []

        for batch_idx, sample in enumerate(train_loader):
            start_time = time.time()
            global_step = len(train_loader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            loss = train_sample(sample, detailed_summary=do_summary)
            this_loss.append(loss)

            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx, len(train_loader), loss, time.time() - start_time))

        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logckptdir, epoch_idx))
            
            print("model_{:0>6}.ckpt saved".format(epoch_idx))
        
        this_loss = np.mean(this_loss)
        print("Epoch loss: {:.5f} --> {:.5f}".format(last_loss, this_loss))

        lr_scheduler.step()

def loss_fn(depth_est, depth_gt, mask, delta):
    loss_depth = (depth_est[mask] - depth_gt[mask]).abs() / (delta.unsqueeze(1).unsqueeze(1) * mask.sum())
    loss_depth = loss_depth.sum()

    return loss_depth

def train_sample(sample, detailed_summary=False):
    """
    :param sample: each batch
    :param detailed_summary: whether the detailed logs are needed.
    :return: the loss
    """
    # model.train() is not needed here, however it is often used to state this script is not for evaluation.
    model.train()
    optimizer.zero_grad()

    if args.cuda:
        sample = tocuda(sample)
    ref_depths = sample["ref_depths"]

    # forward
    outputs = model(sample["ref_img"].float(), sample["src_imgs"].float(), 
                    sample["ref_extrinsics"], sample["src_extrinsics"],
                    sample["depth_values"])

    depth_est_list = outputs["depth_est_list"]

    # print(ref_depths[:, 0])
    # print(depth_est_list[0])
    # exit()

    delta = (sample["depth_max"] - sample["depth_min"]) / (args.ndepths - 1)

    loss = []
    for i in range(ref_depths.shape[1]):
        depth_gt = ref_depths[:, i]
        # print(depth_gt.min(), depth_gt.max())
        # exit()
        mask = torch.logical_and(args.depth_min < depth_gt, depth_gt < args.depth_max)
        loss.append(loss_fn(depth_est_list[i], depth_gt.float(), mask, delta))

    loss = sum(loss)

    loss.backward()

    optimizer.step()

    return loss.data.cpu().item()

train()