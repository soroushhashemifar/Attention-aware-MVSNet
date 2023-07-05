import os
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from datasets.data_io import save_pfm
import cv2

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from model import AttMVSNet
from utils import tocuda, tensor2numpy
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

    epochs = 40
    lr = 0.001
    lrepochs = "10,20,30,40:2"
    wd = 0.0
    batch_size = 5
    summary_freq = 1
    save_freq = 1

    init_epoch = 0
    loadckpt = '/content/ATT_MVSNET/model_000047.ckpt'
    logckptdir = './checkpoints/'
    outdir = "./results/"

    cuda = True


args = Args()

def save_depth():
    # dataset, dataloader
    test_dataset = AttMVSDataset(args)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=16, drop_last=False)

    # model
    model = AttMVSNet(args)
    if args.cuda:
        model.cuda()

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            if batch_idx == 10:
                break

            if args.cuda:
                sample = tocuda(sample)

            outputs = model(sample["ref_img"].float(), sample["src_imgs"].float(), 
                    sample["ref_extrinsics"], sample["src_extrinsics"],
                    sample["depth_values"])
            outputs = tensor2numpy(outputs)
            print('Iter {}/{}'.format(batch_idx, len(TestImgLoader)))
            filenames = sample["filename"]

            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence, ref_image, ref_depth in zip(filenames, outputs["depth_est_list"][0],
                                                                   outputs["photometric_confidence"], sample["ref_img"], sample["ref_depths"]):
                depth_filename = os.path.join(args.outdir, filename+"_depth"+'.pfm')
                confidence_filename = os.path.join(args.outdir, filename+"_conf"+'.pfm')
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # # save depth maps
                # save_pfm(depth_filename, depth_est)
                # # save confidence maps
                # save_pfm(confidence_filename, photometric_confidence)

                ref_depth_filename = os.path.join(args.outdir, filename+"_ref_depth"+'.png')
                ref_depth = ref_depth.permute(1, 2, 0).detach().cpu().numpy()
                ref_depth = (ref_depth - ref_depth.min()) / (ref_depth.max() - ref_depth.min())
                ref_depth = ref_depth * 255
                cv2.imwrite(ref_depth_filename, ref_depth)

                image_filename = os.path.join(args.outdir, filename+"_ref_image"+'.png')
                ref_image = ref_image.permute(1, 2, 0).detach().cpu().numpy()
                ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
                ref_image = ref_image * 255
                cv2.imwrite(image_filename, ref_image)

                depth_filename = os.path.join(args.outdir, filename+"_depth"+'.png')
                depth_est = (depth_est - depth_est.min()) / (depth_est.max() - depth_est.min())
                depth_est = depth_est * 255
                cv2.imwrite(depth_filename, depth_est)

                confidence_filename = os.path.join(args.outdir, filename+"_conf"+'.png')
                photometric_confidence = (photometric_confidence - photometric_confidence.min()) / (photometric_confidence.max() - photometric_confidence.min())
                photometric_confidence = photometric_confidence * 255
                cv2.imwrite(confidence_filename, photometric_confidence)


save_depth()