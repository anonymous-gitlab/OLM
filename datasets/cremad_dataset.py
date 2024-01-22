import os
import csv
import copy
import librosa

import torch
import random
import numpy as np
from scipy import signal
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class CramedDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = '/home/xxx/cv/OLM/data/CREMAD/'
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}
       
        if args.dataset  == 'CREMAD':
            self.audio_feature_path = '/home/xxx/cv/OLM/data/CREMA-D/AudioWAV/'
            self.visual_feature_path = '/home/xxx/cv/OLM/data/CREMA-D/'

        self.train_csv = os.path.join(self.data_root, 'train.csv')
        self.test_csv = os.path.join(self.data_root, 'test.csv')
        self.union_csv = os.path.join(self.data_root, 'union.csv')

        if mode == 'train':
            # csv_file = self.train_csv
            csv_file = self.union_csv
        else:
            csv_file = self.test_csv

    
        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item[0])
                
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        # audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # mean = np.mean(spectrogram)
        # std = np.std(spectrogram)
        # spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # visual
        image_samples = os.listdir(self.image[idx])
        image_samples = np.array(image_samples)
        select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        select_index.sort()
        image_samples = image_samples[select_index]
        images = torch.zeros((self.args.fps, 3, 224, 224))
        for i in range(self.args.fps):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        # images = torch.permute(images, (1,0,2,3))
        images = images.permute((1,0,2,3))

        # label
        label = self.label[idx]        
        return spectrogram, images, label
    

# for five-fold cross-validation on Train&Val
def get_cremad_loaders(args, config):
    train_dataset = CramedDataset(args, mode='train')

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
        test_dataset = CramedDataset(args, mode='test')
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=True)
        test_loaders.append(test_loader)

    ## return loaders
    adim, tdim, vdim = 0,0,0
    return train_loaders, eval_loaders, test_loaders, adim, tdim, vdim