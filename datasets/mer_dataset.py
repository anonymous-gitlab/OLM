import os
import tqdm
import glob
import random
import numpy as np
import pandas as pd
import multiprocessing

import torch
import config
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


emos = ['neutral', 'angry', 'happy', 'sad', 'worried',  'surprise']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos): emo2idx[emo] = ii
for ii, emo in enumerate(emos): idx2emo[ii] = emo


def func_read_one(argv=None, feature_root=None, name=None):
    
    feature_root, name = argv
    feature_dir = glob.glob(os.path.join(feature_root, name+'*'))
    assert len(feature_dir) == 1
    feature_path = feature_dir[0]

    feature = []
    if feature_path.endswith('.npy'):
        single_feature = np.load(feature_path)
        single_feature = single_feature.squeeze()
        feature.append(single_feature)
    else:
        facenames = os.listdir(feature_path)
        for facename in sorted(facenames):
            facefeat = np.load(os.path.join(feature_path, facename))
            feature.append(facefeat)

    single_feature = np.array(feature).squeeze()
    if len(single_feature) == 0:
        print ('feature has errors!!')
    elif len(single_feature.shape) == 2:
        single_feature = np.mean(single_feature, axis=0)
    return single_feature


def read_data_multiprocess(label_path, feature_root, task='emo', data_type='train', debug=False):
    ## gain (names, labels)
    names, labels = [], []
    assert task in  ['emo', 'aro', 'val', 'whole']
    assert data_type in ['train', 'test1', 'test2', 'test3']
    if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
    if data_type == 'test1': corpus = np.load(label_path, allow_pickle=True)['test1_corpus'].tolist()
    if data_type == 'test2': corpus = np.load(label_path, allow_pickle=True)['test2_corpus'].tolist()
    if data_type == 'test3': corpus = np.load(label_path, allow_pickle=True)['test3_corpus'].tolist()
    for name in corpus:
        names.append(name)
        if task in ['aro', 'val']:
            labels.append(corpus[name][task])
        if task == 'emo':
            labels.append(emo2idx[corpus[name]['emo']])
        if task == 'whole':
            corpus[name]['emo'] = emo2idx[corpus[name]['emo']]
            labels.append(corpus[name])

    ## ============= for debug =============
    if debug: 
        names = names[:100]
        labels = labels[:100]
    ## =====================================

    ## names => features
    params = []
    for ii, name in tqdm.tqdm(enumerate(names)):
        params.append((feature_root, name))
    
    
    features = []
    with multiprocessing.Pool(processes=16) as pool:
        features = list(tqdm.tqdm(pool.imap(func_read_one, params), total=len(params)))        
    feature_dim = np.array(features).shape[-1]
    
    ## save (names, features)
    print (f'Input feature {feature_root} ===> dim is {feature_dim}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    name2feats, name2labels = {}, {}
    for ii in range(len(names)):
        name2feats[names[ii]]  = features[ii]
        name2labels[names[ii]] = labels[ii]
    return name2feats, name2labels, feature_dim

class MERDataset(Dataset):
    def __init__(self, label_path, audio_root, text_root, video_root, data_type, debug=False):
        assert data_type in ['train', 'test1', 'test2', 'test3']
        self.name2audio, self.name2labels, self.adim = read_data_multiprocess(label_path, audio_root, task='whole', data_type=data_type, debug=debug)
        self.name2text,  self.name2labels, self.tdim = read_data_multiprocess(label_path, text_root,  task='whole', data_type=data_type, debug=debug)
        self.name2video, self.name2labels, self.vdim = read_data_multiprocess(label_path, video_root, task='whole', data_type=data_type, debug=debug)
        self.names = [name for name in self.name2audio if 1==1]

    def __getitem__(self, index):
        name = self.names[index]
        return torch.FloatTensor(self.name2audio[name]),\
               torch.FloatTensor(self.name2text[name]),\
               torch.FloatTensor(self.name2video[name]),\
               self.name2labels[name]['emo'],\
               self.name2labels[name]['val'],\
               name

    def __len__(self):
        return len(self.names)

    def get_featDim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

## for five-fold cross-validation on Train&Val
def get_loaders(args, config):
    train_dataset = MERDataset(label_path = config.PATH_TO_LABEL[args.train_dataset], # dataset='MER2023'
                               audio_root = os.path.join(config.PATH_TO_FEATURES[args.train_dataset], args.audio_feature),
                               text_root  = os.path.join(config.PATH_TO_FEATURES[args.train_dataset], args.text_feature),
                               video_root = os.path.join(config.PATH_TO_FEATURES[args.train_dataset], args.video_feature),
                               data_type  = 'train',
                               debug      = args.debug)

    # gain indices for cross-validation
    whole_folder = []
    whole_num = len(train_dataset.names)
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
        test_dataset = MERDataset(label_path = config.PATH_TO_LABEL[args.test_dataset], #  test_dataset='MER2023'
                                  audio_root = os.path.join(config.PATH_TO_FEATURES[args.test_dataset], args.audio_feature),
                                  text_root  = os.path.join(config.PATH_TO_FEATURES[args.test_dataset], args.text_feature),
                                  video_root = os.path.join(config.PATH_TO_FEATURES[args.test_dataset], args.video_feature),
                                  data_type  = 'train',
                                  debug      = args.debug)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=False)
        test_loaders.append(test_loader)

    ## return loaders
    adim, tdim, vdim = train_dataset.get_featDim()
    return train_loaders, eval_loaders, test_loaders, adim, tdim, vdim