import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, DriverFocusCrop, DriverCenterCrop)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, UniformRandomSample, UniformEndSample
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set
from utils import Logger
from train import train_epoch
from validation import val_epoch


if __name__ == '__main__':


    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    weights = [1, 2, 4, 2, 4]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center', 'driver focus']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        elif opt.train_crop == 'driver focus':
            crop_method = DriverFocusCrop(opt.scales, opt.sample_size)
        train_spatial_transform = Compose([
            crop_method,
            MultiScaleRandomCrop(opt.scales, opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        train_temporal_transform = UniformRandomSample(opt.sample_duration, opt.end_second)
        train_target_transform = ClassLabel()
        train_horizontal_flip = RandomHorizontalFlip()
        training_data = get_training_set(opt, train_spatial_transform, train_horizontal_flip,
                                         train_temporal_transform, train_target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.lr_step, gamma=0.1)
    if not opt.no_val:
        val_spatial_transform = Compose([
            DriverCenterCrop(opt.scales, opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        val_temporal_transform = UniformEndSample(opt.sample_duration, opt.end_second)
        val_target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, val_spatial_transform, val_temporal_transform, val_target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=24,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):

        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, optimizer, opt,
                                        val_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step()


    # for i, (inputs, targets, video) in enumerate(val_loader):
    #     if i%100 == 0:
    #         print("Processing to image {}".format(i))
    #     name = video[0]
    #     model.eval()

    #     file_path_base = name.find('face_camera',1)
    #     file_path = name[:file_path_base]

    #     s = name[file_path_base+12:]
    #     idx = s.split('/')
    #     file_path = os.path.join(file_path, 'features','face', idx[0])
    #     file_name = idx[1] + '.pt'
    #     save_path = os.path.join(file_path, file_name)

    #     val_out = model(inputs).squeeze().cpu()
    #     print(save_path)
    #     torch.save(val_out, save_path)

    # #      print(save_path)
    # #      b.save(name1,"PNG")
    # #      a.save(name1,'PNG')
    # #    sys.exit()
