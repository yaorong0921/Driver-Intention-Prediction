import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from os.path import *
import numpy as np
import random
from glob import glob
import csv
from utils import load_value_file

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image-{:04d}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print(image_path)
            return video

    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

def load_annotation_data(data_file_path, fold):
    database = {}
    data_file_path = os.path.join(data_file_path, 'fold%d.csv'%fold)
    print('Load from %s'%data_file_path)
    with open(data_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            value = {}
            value['subset'] = row[3]
            value['label'] = row[1]
            value['n_frames'] = int(row[2])
            database[row[0]] = value
    return database

def get_class_labels():
#### define the labels map
    class_labels_map = {}
    class_labels_map['end_action'] = 0
    class_labels_map['lchange'] = 1
    class_labels_map['lturn'] = 2
    class_labels_map['rchange'] = 3
    class_labels_map['rturn'] = 4
    return class_labels_map

def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data.items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['label']
            video_names.append(key)    ### key = 'rturn/20141220_154451_747_897'
            annotations.append(value)

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video, end_second,
                 sample_duration, fold):

    data = load_annotation_data(annotation_path, fold)

    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels()
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 100 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            print('File does not exists: %s'%video_path)
            continue

#        n_frames = annotations[i]['n_frames']
        # count in the dir
        l = os.listdir(video_path)
        # If there are other files (e.g. original videos) besides the images in the folder, please abstract.
        n_frames = len(l)-2

        if n_frames < 16 + 25*(end_second-1):
            print('Video is too short: %s'%video_path)
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,

            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                for j in range(0, n_samples_for_each_video):
                    sample['frame_indices'] = list(range(1, n_frames+1))
                    sample_j = copy.deepcopy(sample)
                    dataset.append(sample_j)
    return dataset, idx_to_class


class Brain4cars_Inside(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold, 
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        h_flip = False

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        
        if self.horizontal_flip is not None:
            p = random.random()
            if p < 0.5:
                h_flip = True
                clip = [self.horizontal_flip(img) for img in clip]

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        if (h_flip == True) and (target != 0):
            if target == 1:
                target = 3
            elif target == 3:
                target = 1
            elif target == 2:
                target = 4
            elif target == 4:
                target = 2

        return clip, target
    def __len__(self):
        return len(self.data)

class Brain4cars_Outside(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold,
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=5,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is an image.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        h_flip = False

        if self.temporal_transform is not None:
            frame_indices,target_idc = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        target = self.loader(path, target_idc)

        if self.horizontal_flip is not None:
            p = random.random()
            if p < 0.5:
                clip = [self.horizontal_flip(img) for img in clip]
                target = [self.horizontal_flip(img) for img in target]

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0)

        if self.target_transform is not None:
            target = [self.target_transform(img) for img in target]
        target = torch.stack(target, 0).permute(1, 0, 2, 3).squeeze()
            

        return clip, target
    def __len__(self):
        return len(self.data)