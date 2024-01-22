import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random
# from petrel_client.client import Client
import io
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# class MyClient(object):
#     def __init__(self):
#         self.client = Client()
#         print('VGG feature client created.')
#         now = datetime.now()
#         current_time = now.strftime("%Y-%m-%d %H:%M:%S")
#         print(f'{current_time} creat dataloader client')
        
#     def get(self, key):
#         index = key.find("/")  # 桶/key
#         bucket = key[:index]
#         key = key[index+1:]
#         if bucket == "BJ16ER":
#             return self.client.get("BJ16ER:s3://"+ key)
#         elif bucket == "ASR":
#             return self.client.get("ASR:s3://ASR/"+ key)
#         elif bucket == "ASR2":
#             return self.client.get("ASR2:s3://ASR2/"+ key)
#         elif bucket == "emotion-data":
#             return self.client.get("1988:s3://emotion-data/"+ key)
#         else:
#             raise ValueError('wrong key value')

# Define a class to manage the Petrel Client
class MyClient(object):    
    def __init__(self):
        self.client = Client()
        print('VGG feature client created.')
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f'{current_time} creating dataloader client')

    def get(self, key):
        buckets = {
            "BJ16ER": "BJ16ER:s3://",
            "ASR": "ASR:s3://ASR/",
            "ASR2": "ASR2:s3://ASR2/",
            "emotion-data": "1988:s3://emotion-data/"
        }
        bucket_name, object_key = key.split("/", 1)
        if bucket_name in buckets:
            return self.client.get(f'{buckets[bucket_name]}{object_key}')
        else:
            raise ValueError('Invalid bucket name')

def gain_bit_data_from_client(waveforms_obj, client):
    # if waveforms_obj[:3] != 'ASR':
    #     key = 'emotion-data/' + waveforms_obj
    # else:
    #     key = waveforms_obj
    key = waveforms_obj
    b = client.get(key)
    assert b!= None, print(key)
    return b


class VGGSound(Dataset):
    ''' VGGSound is a large-scale video dataset that contains 309 classes, covering a wide range of audio events in everyday life. In our experimental settings, 168,618 videos are used for training and validation, and 13,954 videos are used for testing because some videos are not available now on YouTube.
    paper: Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 721-725. IEEE, 2020
    '''
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        train_video_data = []
        train_audio_data = []
        test_video_data  = []
        test_audio_data  = []
        train_label = []
        test_label  = []
        train_class = []
        test_class  = []
        self.myclient = MyClient()

        with open('/home/xxx/cv/OLM/MML/VGGSound_processed/pkl_list.txt', 'r') as f:
            lines = f.read().splitlines()
        
        pkl_keys = set(lines)
        
        with open('/home/xxx/cv/OLM/data/VGGSound/vggsound.csv') as f:
            csv_reader = csv.reader(f)

            lens = 199176
            for index, item in enumerate(csv_reader):
                if index % 10000 == 0:
                    print(f'samples processing: {index}/{lens}')
                if item[3] == 'train':
                    sample_name = item[0]+'_' + f'{int(item[1]):06d}'
                    if sample_name in pkl_keys:
                        video_dir = sample_name
                        audio_dir = sample_name
                        train_video_data.append(video_dir)
                        train_audio_data.append(audio_dir)
                        if item[2] not in train_class: train_class.append(item[2])
                        train_label.append(item[2])

                if item[3] == 'test':
                    sample_name = item[0]+'_' + f'{int(item[1]):06d}'
                    if sample_name in pkl_keys:
                        video_dir = sample_name
                        audio_dir = sample_name
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)
                        if item[2] not in test_class: test_class.append(item[2])
                        test_label.append(item[2])

                # ! reduce dataset samples
                if args.debug == True:
                    if index >= 10000:
                        break

        # cost lots of time
        print(f'len train_data: {len(train_video_data)}; len test_data: {len(test_video_data)}')
        # assert len(train_class) == len(test_class), print(f'len(train_class): {len(train_class)}, len(test_class): {len(test_class)}')  
        self.classes = train_class

        class_dict = dict(zip(self.classes, range(len(self.classes))))

        if mode == 'train':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[train_label[idx]] for idx in range(len(train_label))]
        if mode == 'test':
            self.video = test_video_data
            self.audio = test_audio_data
            self.label = [class_dict[test_label[idx]] for idx in range(len(test_label))]


    def __len__(self):
        return len(self.video)


    def __getitem__(self, idx):
        feat_path = 'BJ16ER/MML/VGGSound_processed/feats/' + self.video[idx] + '.pkl'
        spectrogram, images, label = pickle.load(io.BytesIO(gain_bit_data_from_client(feat_path, self.myclient))) 
        # load npy files
        # label_1 = self.label[idx]
        # assert label == label_1, 'label mismatch'
        return spectrogram, images, label


# for five-fold cross-validation on Train&Val
def get_vggsound_loaders(args, config):
    train_dataset = VGGSound(args, mode='train')
    
    # gain indices for cross-validation
    whole_folder = []
    whole_num = len(train_dataset)
    indices = np.arange(whole_num)
    random.shuffle(indices)

    # split indices into five-fold
    num_folder = args.num_folder
    each_folder_num = int(whole_num / num_folder)
    for ii in range(num_folder-1):
        each_folder = indices[each_folder_num*ii: each_folder_num*(ii+1)]
        whole_folder.append(each_folder)
    each_folder = indices[each_folder_num*(num_folder-1):]
    whole_folder.append(each_folder)
    assert len(whole_folder) == num_folder
    assert sum([len(each) for each in whole_folder if 1==1]) == whole_num

    ## split into train/eval
    train_eval_idxs = []
    for ii in range(num_folder):
        eval_idxs = whole_folder[ii]
        train_idxs = []
        for jj in range(num_folder):
            if jj != ii: train_idxs.extend(whole_folder[jj])
        train_eval_idxs.append([train_idxs, eval_idxs])

    ## gain train and eval loaders
    train_loaders = []
    eval_loaders = []
    for ii in range(len(train_eval_idxs)):
        train_idxs = train_eval_idxs[ii][0]
        eval_idxs  = train_eval_idxs[ii][1]
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=SubsetRandomSampler(train_idxs),
                                  num_workers=args.num_workers,
                                  pin_memory=True)
        eval_loader = DataLoader(train_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SubsetRandomSampler(eval_idxs),
                                 num_workers=args.num_workers,
                                 pin_memory=True)
        train_loaders.append(train_loader)
        eval_loaders.append(eval_loader)
    
    test_loaders = []
    for test_set in args.test_sets:
        test_dataset = VGGSound(args, mode='test')
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=True)
        test_loaders.append(test_loader)

    ## return loaders
    adim, tdim, vdim = 0,0,0
    return train_loaders, eval_loaders, test_loaders, adim, tdim, vdim


class VGGSound_test(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.myclient = MyClient()
        self.load_data()

    def load_data(self):
        train_video_data, train_audio_data, test_video_data, test_audio_data = [], [], [], []
        train_label, test_label = [], []
        train_class, test_class = [], []

        with open('/home/xxx/cv/OLM/MML/VGGSound_processed/pkl_list.txt', 'r') as f:
            pkl_keys = set(f.read().splitlines())

        with open('/home/xxx/cv/OLM/data/VGGSound/vggsound.csv') as f:
            csv_reader = csv.reader(f)
            lens = 199176

            for index, item in enumerate(csv_reader):
                if index % 10000 == 0:
                    print(f'samples processing: {index}/{lens}')

                sample_name = item[0] + '_' + f'{int(item[1]):06d}'

                if sample_name in pkl_keys:
                    video_dir = sample_name
                    audio_dir = sample_name

                    if item[3] == 'train':
                        train_video_data.append(video_dir)
                        train_audio_data.append(audio_dir)

                        if item[2] not in train_class:
                            train_class.append(item[2])

                        train_label.append(item[2])

                    if item[3] == 'test':
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)

                        if item[2] not in test_class:
                            test_class.append(item[2])

                        test_label.append(item[2])

                if self.args.debug and index >= 10000:
                    break

        print(f'len train_data: {len(train_video_data)}; len test_data: {len(test_video_data)}')
        self.classes = train_class

        class_dict = dict(zip(self.classes, range(len(self.classes))))

        if self.mode == 'train':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[train_label[idx]] for idx in range(len(train_label))]
        if self.mode == 'test':
            self.video = test_video_data
            self.audio = test_audio_data
            self.label = [class_dict[test_label[idx]] for idx in range(len(test_label))]

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        feat_path = 'BJ16ER/MML/VGGSound_processed/feats/' + self.video[idx] + '.pkl'
        spectrogram, images, label = pickle.load(io.BytesIO(self.myclient.get(feat_path)))
        return spectrogram, images, label
    
def get_vggsound_loaders_test(args, config):
    def create_loader(dataset, sampler):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

    train_dataset = VGGSound(args, mode='train')
    whole_num = len(train_dataset)
    indices = np.arange(whole_num)
    random.shuffle(indices)

    num_folder = args.num_folder
    each_folder_num = int(whole_num / num_folder)
    whole_folder = [indices[each_folder_num * ii: each_folder_num * (ii + 1)] for ii in range(num_folder - 1)]
    whole_folder.append(indices[each_folder_num * (num_folder - 1):])

    train_eval_idxs = []
    for ii in range(num_folder):
        eval_idxs = whole_folder[ii]
        train_idxs = []
        for jj in range(num_folder):
            if jj != ii:
                train_idxs.extend(whole_folder[jj])
        train_eval_idxs.append([train_idxs, eval_idxs])

    train_loaders = [create_loader(train_dataset, SubsetRandomSampler(train_eval_idxs[ii][0])) for ii in range(len(train_eval_idxs))]
    eval_loaders = [create_loader(train_dataset, SubsetRandomSampler(train_eval_idxs[ii][1])) for ii in range(len(train_eval_idxs))]

    test_loaders = []
    for test_set in args.test_sets:
        test_dataset = VGGSound(args, mode='test')
        test_loaders.append(create_loader(test_dataset, sampler=None))

    adim, tdim, vdim = 0, 0, 0
    return train_loaders, eval_loaders, test_loaders, adim, tdim, vdim
