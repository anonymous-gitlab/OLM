import os
import csv
import copy
import torch
import random
import librosa
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class AVEDataset(Dataset):
    '''
    AVE is an audio-visual video dataset for audio-visual event localization, 
    which covers 28 event classes and consists of 4,143 10-second videos with both auditory and visual tracks as well as frame-level annotations. All videos are collected from YouTube. 
    '''
    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.args = args
        self.my_dict = {
                0: "Violin-fiddle",
                1: "Female_speech-woman_speaking",
                2: "Toilet_flush",
                3: "Baby_cry-infant_cry",
                4: "Bark",
                5: "Church_bell",
                6: "Race_car-auto_racing",
                7: "Male_speech-man_speaking",
                8: "Horse",
                9: "Train_horn",
                10: "Acoustic_guitar",
                11: "Fixed-wing_aircraft-airplane",
                12: "Accordion",
                13: "Truck",
                14: "Goat",
                15: "Clock",
                16: "Cat",
                17: "Chainsaw",
                18: "Motorcycle",
                19: "Mandolin",
                20: "Ukulele",
                21: "Helicopter",
                22: "Frying_(food)",
                23: "Rodents-rats-mice",
                24: "Shofar",
                25: "Bus",
                26: "Banjo",
                27: "Flute"}
        if args.dataset == 'AVE':
            self.data_root = '/home/xxx/cv/OLM/data/AVE_proessed/AVE/'
            
        self.visual_feature_path = os.path.join(self.data_root, 'visual/')
        self.audio_path = os.path.join(self.data_root, 'audio/') # audio_spec
        self.stat_path = os.path.join(self.data_root, 'stat.txt')
        self.train_txt = os.path.join(self.data_root, 'my_train.txt')
        self.union_txt = os.path.join(self.data_root, 'my_union.txt')
        self.test_txt = os.path.join(self.data_root, 'my_test.txt')

        with open(self.stat_path) as f1:
            csv_reader = csv.reader(f1)
            for row in csv_reader:
                classes.append(row[0])

        if mode == 'train':
            csv_file = self.union_txt
        else:
            csv_file = self.test_txt

        with open(csv_file) as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_path, item[1] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, item[1])
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if args.dataset == 'AVE':
                        # ave, delete repeated labels
                        a = set(data)
                        if item[1] in a:
                            del data2class[item[1]]
                            data.remove(item[1])
                    data.append(item[1])
                    data2class[item[1]] = item[0]
                else:
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

    def __len__(self):
        return len(self.av_files)

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        audio_path = os.path.join(self.audio_path, av_file + '.wav')

        samples, rate = librosa.load(audio_path, sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]

        
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

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
        visual_path = os.path.join(self.visual_feature_path, av_file)
        image_samples = os.listdir(visual_path)
        
        num_samples = len(image_samples)
        choice_nums = min(num_samples, self.args.fps)
        
        select_index = np.random.choice(len(image_samples), size=choice_nums, replace=False)
        select_index.sort()
        
        # select_samples = image_samples
        select_samples = np.array(image_samples)
        select_samples = select_samples[select_index]
        
        images = torch.zeros((choice_nums, 3, 224, 224))
        for i in range(choice_nums):
            img = Image.open(os.path.join(visual_path, select_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        # images = torch.permute(images, (1,0,2,3))
        images = images.permute((1,0,2,3))

        # label
        label = self.classes.index(self.data2class[av_file])
        return spectrogram, images, label
    

## for five-fold cross-validation on Train&Val
def get_ave_loaders(args, config):
    train_dataset = AVEDataset(args, mode='train')

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
                                  pin_memory=False)
        eval_loader = DataLoader(train_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SubsetRandomSampler(eval_idxs),
                                 num_workers=args.num_workers,
                                 pin_memory=False)
        train_loaders.append(train_loader)
        eval_loaders.append(eval_loader)
    
    test_loaders = []
    for test_set in args.test_sets:
        test_dataset = AVEDataset(args, mode='test')
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=False)
        test_loaders.append(test_loader)

    ## return loaders
    adim, tdim, vdim = 0,0,0
    return train_loaders, eval_loaders, test_loaders, adim, tdim, vdim