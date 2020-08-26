import os
import sys
import time
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from math import log10

import torchvision.transforms.functional as F

from opts import parse_opts
from mean import get_mean, get_std
from spatial_transforms import (
	Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
	MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, DriverFocusCrop, DriverCenterCrop)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, UniformIntervalCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from models.convolution_lstm import encoder
from utils import AverageMeter, Logger
from dataset import get_training_set, get_validation_set
import pytorch_ssim

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
	opt.arch = 'ConvLSTM'
	opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
	opt.std = get_std(opt.norm_value)
	print(opt)
	with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
		json.dump(vars(opt), opt_file)

	torch.manual_seed(opt.manual_seed)

### convlstm ##########################################################################################################
	model = encoder(hidden_channels=[128, 64, 64, 32], sample_size=opt.sample_size, sample_duration=opt.sample_duration).cuda()

	model = nn.DataParallel(model, device_ids=None)
	parameters = model.parameters()
	print(model)

	criterion = nn.MSELoss()
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
		Scale(opt.sample_size),		
		ToTensor(opt.norm_value) #, norm_method
	])
	train_temporal_transform = UniformIntervalCrop(opt.sample_duration, opt.interval)
	train_target_transform = Compose([
		Scale(opt.sample_size),
		ToTensor(opt.norm_value)#, norm_method
	])
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
		os.path.join(opt.result_path, 'convlstm-train.log'),
		['epoch', 'loss', 'lr'])
	train_batch_logger = Logger(
		os.path.join(opt.result_path, 'convlstm-train_batch.log'),
		['epoch', 'batch', 'iter', 'loss', 'lr'])

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
			Scale(opt.sample_size),
			ToTensor(opt.norm_value)#, norm_method
		])
		val_temporal_transform = UniformIntervalCrop(opt.sample_duration, opt.interval)
		val_target_transform = val_spatial_transform
		validation_data = get_validation_set(
		    opt, val_spatial_transform, val_temporal_transform, val_target_transform)
		val_loader = torch.utils.data.DataLoader(
			validation_data,
			batch_size=1,
			shuffle=True,
			num_workers=opt.n_threads,
			pin_memory=True)
		val_logger = Logger(
			os.path.join(opt.result_path, 'convlstm-val.log'), ['epoch', 'loss', 'ssim', 'psnr'])

	if opt.resume_path:
		print('loading checkpoint {}'.format(opt.resume_path))
		checkpoint = torch.load(opt.resume_path)
		assert opt.arch == checkpoint['arch']

		opt.begin_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		if not opt.no_train:
			optimizer.load_state_dict(checkpoint['optimizer'])
#===============================================================================================
	print('run')
	global best_loss
	best_loss = torch.tensor(float('inf'))

	for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
		if not opt.no_train:
			print('train at epoch {}'.format(epoch))

			model.train()

			batch_time = AverageMeter()
			data_time = AverageMeter()
			losses = AverageMeter()

			end_time = time.time()
			for i, (inputs, targets) in enumerate(train_loader):
				data_time.update(time.time() - end_time)

				if not opt.no_cuda:
					targets = targets.cuda(non_blocking=True)
				inputs = Variable(inputs)
				targets = Variable(targets)
				outputs = model(inputs)
				loss = criterion(outputs, targets)

				losses.update(loss.item(), inputs.size(0))

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
					'lr': optimizer.param_groups[0]['lr']
				})
				if i % 50 == 0:
					print('Epoch: [{0}][{1}/{2}]\t'
						  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
						  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
							  epoch,
							  i + 1,
							  len(train_loader),
							  batch_time=batch_time,
							  data_time=data_time,
							  loss=losses))

			train_logger.log({
				'epoch': epoch,
				'loss': losses.avg,
				'lr': optimizer.param_groups[0]['lr']
			})

			if epoch % opt.checkpoint == 0:
				save_file_path = os.path.join(opt.result_path,
											  'convlstm-save_{}.pth'.format(epoch))
				states = {
					'epoch': epoch + 1,
					'arch': opt.arch,
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict(),
				}

		if not opt.no_val:
			model.eval()

			batch_time = AverageMeter()
			data_time = AverageMeter()
			losses = AverageMeter()

			ssim_losses = AverageMeter()
			psnr_losses = AverageMeter()

			ssim_criterion = pytorch_ssim.SSIM(window_size = 11, size_average = True)

			end_time = time.time()


			for i, (inputs, targets) in enumerate(val_loader):
				data_time.update(time.time() - end_time)

				if not opt.no_cuda:
					targets = targets.cuda(non_blocking=True)
				inputs = Variable(inputs, volatile=True)
				targets = Variable(targets, volatile=True)
				outputs = model(inputs)
				loss = criterion(outputs, targets)

				ssim = ssim_criterion(outputs,targets)
				psnr = 10 * log10(1 / loss.item())

				losses.update(loss.item(), inputs.size(0))
				ssim_losses.update(ssim.item(), inputs.size(0))
				psnr_losses.update(psnr, inputs.size(0))

				batch_time.update(time.time() - end_time)
				end_time = time.time()

				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'SSIM {ssim_loss.val:.4f}({ssim_loss.avg:.4f})\t'
					  'PSNR {psnr_loss.val:.4f}({psnr_loss.avg:.4f})'.format(
						  epoch,
						  i + 1,
						  len(val_loader),
						  batch_time=batch_time,
						  data_time=data_time,
						  loss=losses,
						  ssim_loss=ssim_losses,
						  psnr_loss=psnr_losses))

			val_logger.log({'epoch': epoch, 'loss': losses.avg, 'ssim': ssim_losses.avg, 'psnr': psnr_losses.avg})

			is_best = losses.avg < best_loss
			best_loss = min(losses.avg, best_loss)
			print('\n The best prec is %.4f' % best_loss)
			if is_best:
				states = {
					'epoch': epoch + 1,
					 'arch': opt.arch,
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict(),
				  }
				save_file_path = os.path.join(opt.result_path,
									'convlstm-save_best.pth')
				torch.save(states, save_file_path)


		if not opt.no_train and not opt.no_val:
			scheduler.step()




