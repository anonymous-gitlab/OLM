""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional

import re
import os
import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .readers import create_reader

_logger = logging.getLogger(__name__)

_ERROR_RETRY = 50

class ThroatDataset(data.Dataset):
    def __init__(self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        try:
            split in ['train','test','val']
            if split=='train':
                self.root_path = root+'/mirror_train/'
            elif split=='val' or split == 'test':
                self.root_path = root+'/mirror_test/'
        except AssertionError as e:
            _logger.info("Check your dataset train/val split info")
            raise e

        self.samples = []
        self.nb_classes = 5 
        self.labels = []
        self.samples_per_cls = np.zeros(self.nb_classes)

        self.load_bytes = load_bytes
        self.img_mode = img_mode
        # self.transform overloaded in the create_loader function
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

        # for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        for root_dir, subdirs, files in os.walk(self.root_path, topdown=False,followlinks=True):
            for name in files:
                matchObj = re.match('(.*)Store', name, re.M|re.I)
                if matchObj is not None:
                    os.remove(os.path.join(root_dir, name))
                    continue
                    # print(os.path.join(root, name))    
                current_file = os.path.join(root_dir,name)
                if re.match('(.*)group2', root_dir, re.M | re.I) is not None:
                    if 'a' in root_dir.split('/')[-1]:
                        current_label = 1
                    elif 'b' in root_dir.split('/')[-1]:
                        current_label = 2 
                    elif 'c' in root_dir.split('/')[-1]:
                        current_label = 3
                    else:
                        raise ValueError('Invalid root directory: %s' % root_dir)
                elif re.match('(.*)group1', root_dir, re.M|re.I) is not None:
                    current_label = 0
                elif re.match('(.*)group3', root_dir, re.M|re.I) is not None:
                    current_label = 4
                assert current_label in [0, 1, 2, 3, 4], "Invalid target, check your dataset"
                self.samples.append((current_file,current_label))
                self.labels.append(current_label)
        
        # # log print dataset statistics
        _logger.info(f"====> Number of total instances is {len(self.samples)}")
        _logger.info(f"==> Number of class 0 is {len([item for item in self.samples if item[1]==0])}") # Benign lesions
        _logger.info(f"==> Number of class 1 is {len([item for item in self.samples if item[1]==1])}") # leukoplakia (mid)
        _logger.info(f"==> Number of class 2 is {len([item for item in self.samples if item[1]==2])}") # leukoplakia (moderate)
        _logger.info(f"==> Number of class 3 is {len([item for item in self.samples if item[1]==3])}") # leukoplakia (severe)
        _logger.info(f"==> Number of class 4 is {len([item for item in self.samples if item[1]==4])}") # caner

        for lab in range(self.nb_classes):
            self.samples_per_cls[lab]= self.labels.count(lab)
        _logger.info(f"==> samples_per_cls is {self.samples_per_cls}")

        # class_weight = np.zeros(self.nb_classes)
        # for lab in range(self.nb_classes):
        #     class_weight[lab] = np.sum(self.labels[:] == lab)
        # class_weight = 1 / class_weight
        # self.sample_weight = np.zeros(len(self.labels))
        # for i in range(len(self.labels)):
        #     self.sample_weight[i] = class_weight[self.labels[i]]
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = open(path, 'rb')
        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                    return self.__getitem__((index + 1) % len(self.reader))
            else:
                    raise e
        self._consecutive_errors = 0

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, path
    
    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):

        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename        

class SimpleThroatDataset(data.Dataset):
    def __init__(self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        try:
            split in ['train','test','val']
            if split=='train':
                self.root_path = root+'/train/'
            elif split=='val' or split == 'test':
                self.root_path = root+'/test/'
        except AssertionError as e:
            _logger.info("Check your dataset train/val split info")
            raise e

        self.samples = []
        self.nb_classes = 3
        self.labels = []
        self.samples_per_cls = np.zeros(self.nb_classes)

        self.load_bytes = load_bytes
        self.img_mode = img_mode
        # self.transform overloaded in the create_loader function
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

        # for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        for root_dir, subdirs, files in os.walk(self.root_path, topdown=False,followlinks=True):
            for name in files:
                matchObj = re.match('(.*)Store', name, re.M|re.I)
                if matchObj is not None:
                    os.remove(os.path.join(root_dir, name))
                    continue
                    # print(os.path.join(root, name))    
                current_file = os.path.join(root_dir,name)
                if re.match('(.*)group2', root_dir, re.M | re.I) is not None:
                    if 'a' in root_dir.split('/')[-1]:
                        current_label = 1
                    elif 'b' in root_dir.split('/')[-1]:
                        current_label = 1 
                    elif 'c' in root_dir.split('/')[-1]:
                        current_label = 1
                    else:
                        raise ValueError('Invalid root directory: %s' % root_dir)
                elif re.match('(.*)group1', root_dir, re.M|re.I) is not None:
                    current_label = 0
                elif re.match('(.*)group3', root_dir, re.M|re.I) is not None:
                    current_label = 2
                assert current_label in [0, 1, 2], "Invalid target, check your dataset"
                self.samples.append((current_file,current_label))
                self.labels.append(current_label)

        # log print dataset statistics
        _logger.info(f"====> Number of total instances is {len(self.samples)}")
        _logger.info(f"==> Number of class 0 (Benign lesions) is {len([item for item in self.samples if item[1]==0])}") # Benign lesions
        # _logger.info(f"==> Number of class 1 is {len([item for item in self.samples if item[1]==1])}") # leukoplakia (mid)
        # _logger.info(f"==> Number of class 2 is {len([item for item in self.samples if item[1]==2])}") # leukoplakia (moderate)
        # _logger.info(f"==> Number of class 3 is {len([item for item in self.samples if item[1]==3])}") # leukoplakia (severe)
        _logger.info(f"==> Number of class 1 (Leukoplakia) is {len([item for item in self.samples if item[1]==1])}") # leukoplakia (severe)
        _logger.info(f"==> Number of class 2 (Cancer) is {len([item for item in self.samples if item[1]==2])}") # cancer
         
        for lab in range(self.nb_classes):
            self.samples_per_cls[lab]= self.labels.count(lab)
        _logger.info(f"==> samples_per_cls is {self.samples_per_cls}")

        
        # class_weight = np.zeros(self.nb_classes)
        # for lab in range(self.nb_classes):
        #     class_weight[lab] = np.sum(self.labels[:] == lab)
        # class_weight = 1 / class_weight
        # self.sample_weight = np.zeros(len(self.labels))
        # for i in range(len(self.labels)):
        #     self.sample_weight[i] = class_weight[self.labels[i]]
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = open(path, 'rb')
        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                    return self.__getitem__((index + 1) % len(self.reader))
            else:
                    raise e
        self._consecutive_errors = 0

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, path
    
    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):

        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename        

class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]
        
        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            is_training=False,
            batch_size=None,
            seed=42,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                is_training=is_training,
                batch_size=batch_size,
                seed=seed,
                repeats=repeats,
                download=download,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        super(AugMixDataset, self).__init__()
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y, x_path = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y, x_path

    def __len__(self):
        return len(self.dataset)
