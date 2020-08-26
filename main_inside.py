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

from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy

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
    global best_prec
    best_prec = 0
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):

        if not opt.no_train:
            print('train at epoch {}'.format(epoch))

            model.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()

            end_time = time.time()
            for i, (inputs, targets) in enumerate(train_loader):
                data_time.update(time.time() - end_time)

                if not opt.no_cuda:
                    targets = targets.cuda(non_blocking=True)
                inputs = Variable(inputs)
                targets = Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)

                losses.update(loss.item(), inputs.size(0))
                accuracies.update(acc, inputs.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                train_batch_logger.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * len(train_loader) + (i + 1),
                    'loss': losses.val,
                    'acc': accuracies.val,
                    'lr': optimizer.param_groups[0]['lr']
                })
                if i % 5 == 0:
                  print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                          epoch,
                          i + 1,
                          len(train_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))

            train_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

            if epoch % opt.checkpoint == 0:
                save_file_path = os.path.join(opt.result_path,
                                              'save_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

        if not opt.no_val:
            print('Validation at epoch {}'.format(epoch))

            model.eval()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()

            end_time = time.time()

            conf_mat = torch.zeros(opt.n_finetune_classes, opt.n_finetune_classes)
            output_file = []

            for i, (inputs, targets) in enumerate(val_loader):
                data_time.update(time.time() - end_time)

                if not opt.no_cuda:
                    targets = targets.cuda(non_blocking=True)
                inputs = Variable(inputs, volatile=True)
                targets = Variable(targets, volatile=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)

                ### print out the confusion matrix
                _,pred = torch.max(outputs,1)
                for t,p in zip(targets.view(-1), pred.view(-1)):
                    conf_mat[t,p] += 1

                losses.update(loss.item(), inputs.size(0))
                accuracies.update(acc, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                          epoch,
                          i + 1,
                          len(val_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))
            print(conf_mat)

            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

            is_best = accuracies.avg > best_prec
            best_prec = max(accuracies.avg, best_prec)
            print('\n The best prec is %.4f' % best_prec)
            if is_best:
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                 }
                save_file_path = os.path.join(opt.result_path,
                                    'save_best.pth')
                torch.save(states, save_file_path)

        if not opt.no_train and not opt.no_val:
            scheduler.step()



