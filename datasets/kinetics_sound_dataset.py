import os
import csv
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image

class KSoundsDataset(Dataset):
    '''Kinetics-Sounds (KS) is a dataset containing 31 human action classes selected from Kinetics dataset which contains 400 classes of YouTube videos. Original paper: Arandjelovic, Relja, and Andrew Zisserman. "Look, listen and learn." Proceedings of the IEEE international conference on computer vision. 2017.
    Args:
        args (Namespace): Command-line arguments.
        mode (str): Dataset mode, 'train' or 'test'.
    '''    
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.data_root = './data/'
        self.visual_feature_path = '/home/xxx/cv/OLM/MML/k31'
        self.audio_feature_path = os.path.join(self.visual_feature_path, 'audio_train' if mode == 'train' else 'audio_test')

        csv_file = os.path.join(self.data_root, args.dataset, f'my_{mode}.txt')
        ignore_list = ['BnAfszjLPYI_000003_000013']
        self.image, self.audio, self.label = [], [], []

        with open(csv_file, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if item[0] in ignore_list:
                    continue
                audio_path = os.path.join(self.audio_feature_path, f'{item[0]}.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-01-FPS', item[0])
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(int(item[-1]))
                else:
                    continue

                if args.debug and len(self.image) >= 10000:
                    break

        print(f'{mode}, len data image: {len(self.image)}')

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        # Load and preprocess audio data
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050 * 3]
        resamples = np.clip(resamples, -1, 1)
        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        if spectrogram.shape[1] != 188:
            pad_nums = 188 - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_nums)))

        # Define image transformations
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

        # Load and preprocess visual data
        image_samples = os.listdir(self.image[idx])
        image_samples = np.array(image_samples)
        if len(image_samples) >= self.args.fps:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        else:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=True)
        select_index.sort()
        images = torch.zeros((self.args.fps, 3, 224, 224))
        for i in range(self.args.fps):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img
        images = images.permute((1, 0, 2, 3))

        label = self.label[idx]
        return spectrogram, images, label

def get_ks_loaders(args, config):
    '''
    Create data loaders for the KSounds dataset.

    Args:
        args (Namespace): Command-line arguments.
        config: Configuration parameters.

    Returns:
        tuple: Train, eval, and test data loaders, and dimensions of audio, text, and visual features.
    '''
    train_dataset = KSoundsDataset(args, mode='train')
    
    # Gain indices for cross-validation
    whole_num = len(train_dataset)
    indices = np.arange(whole_num)
    random.shuffle(indices)

    num_folder = args.num_folder
    each_folder_num = int(whole_num / num_folder)
    whole_folder = [indices[each_folder_num * ii: each_folder_num * (ii + 1)] for ii in range(num_folder)]

    train_eval_idxs = []
    for ii in range(num_folder):
        eval_idxs = whole_folder[ii]
        train_idxs = [jj for jj in range(num_folder) if jj != ii]
        train_idxs = [item for sublist in [whole_folder[jj] for jj in train_idxs] for item in sublist]
        train_eval_idxs.append([train_idxs, eval_idxs])

    train_loaders = []
    eval_loaders = []
    for ii in range(len(train_eval_idxs)):
        train_idxs = train_eval_idxs[ii][0]
        eval_idxs = train_eval_idxs[ii][1]
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
        test_dataset = KSoundsDataset(args, mode='test')
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=True)
        test_loaders.append(test_loader)

    adim, tdim, vdim = 0, 0, 0
    return train_loaders, eval_loaders, test_loaders, adim, tdim, vdim
