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
import torchvision

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


def video_loader(image_path,  image_loader):
	video = []
#        image_path = os.path.join(video_dir_path, 'image-{:04d}.png'.format(i))
	if os.path.exists(image_path):
#        video.append(image_loader(image_path))
		video = image_loader(image_path)
	else:
		print(image_path)
	return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def get_default_feature_loader(feature_path):
    feature_tensor = torch.load(feature_path)
    feature_tensor.requires_grad = False

    return feature_tensor

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
#            if subset == 'testing':
#                video_names.append('test/{}'.format(key))
#            else:
            label = value['label']
#            video_names.append('{}/{}'.format(label, key))
            video_names.append(key[:-1])    ### key = 'rturn/20141220_154451_747_897/'
            annotations.append(value)

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, fold):

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

        face_feature_path = os.path.join(root_path, 'features', 'face', '4_1sec', video_names[i]+'.pt')
#        road_feature_path = os.path.join(root_path, 'features', 'road_images', video_names[i]+'.png')
        road_feature_path = os.path.join(root_path, 'features', 'road', '4_first', video_names[i]+'.pt')
#        file_name = os.path.join(road_video_path,'predict-02.png')

        if not (os.path.exists(face_feature_path) and os.path.exists(road_feature_path)):
            print(road_feature_path)
            continue

#        n_frames_file_path = os.path.join(video_path, 'n_frames')
#        n_frames = int(load_value_file(n_frames_file_path))
#        n_frames = annotations[i]['n_frames']
#        if n_frames < 125:
#            print(face_feature_path)
#            continue

#        begin_t = 1
#        end_t = n_frames
        sample = {
            'face_feature': face_feature_path,
            'road_feature': road_feature_path,
            'subset': subset,
#            'segment': [begin_t, end_t],
#            'n_frames': n_frames,
#            'road_video': road_video_path,
#            'video_id': video_names[i][:-14].split('/')[1]
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

#        if n_samples_for_each_video == 1:
#            sample['frame_indices'] = list(range(1, n_frames + 1))
        dataset.append(sample)
#        else:
#            if n_samples_for_each_video > 1:
#                step = max(1,
#                           math.ceil((n_frames - 1 - sample_duration) /
#                                     (n_samples_for_each_video - 1)))
#            else:
#                step = sample_duration
#            for j in range(1, n_frames, step):
#                sample_j = copy.deepcopy(sample)
#                sample_j['frame_indices'] = list(
#                    range(j, min(n_frames + 1, j + sample_duration)))
#                dataset.append(sample_j)
    return dataset, idx_to_class


class Brain4cars_Features(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, nfold)
        self.loader = get_default_video_loader()
#        self.spatial_transform = spatial_transform
#        self.temporal_transform = temporal_transform
#        self.target_transform = target_transform
#        self.horizontal_flip = horizontal_flip

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        face_feature_path = self.data[index]['face_feature']
        face_feature = get_default_feature_loader(face_feature_path)
        target = self.data[index]['label']
        std, mean = torch.std_mean(face_feature)
        face_feature = (face_feature-mean)/std

        road_feature_path = self.data[index]['road_feature']
#        road_feature = self.loader(road_feature_path)
        road_feature = get_default_feature_loader(road_feature_path)
        return face_feature, road_feature, target

    def __len__(self):
        return len(self.data)
