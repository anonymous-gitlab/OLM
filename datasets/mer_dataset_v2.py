import os
import glob
import numpy as np
import random
import multiprocess as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
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
        # single_feature = single_feature.squeeze()
        feature.append(single_feature)
    else:
        facenames = os.listdir(feature_path)
        for facename in sorted(facenames):
            facefeat = np.load(os.path.join(feature_path, facename))
            feature.append(facefeat)

    single_feature = np.array(feature).squeeze(0)
    if len(single_feature) == 0:
        print ('feature has errors!!')
    # elif len(single_feature.shape) == 2:
    #     single_feature = np.mean(single_feature, axis=0)
    return single_feature


def read_data_multiprocess(label_path, feature_root, task='emo', data_type='train', debug=False):
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

    if 'resnet' in feature_root.lower():
        feature_dim = 512
    elif 'manet' in feature_root.lower():
        feature_dim = 1024
    elif 'macbert-large' in feature_root.lower():
        feature_dim = 1024
    elif 'hubert-large' in feature_root.lower():
        feature_dim = 1024
    else:
        raise Exception("Not Implemented Error")
    
    print (f'Input feature {feature_root} ===> dim is {feature_dim}')
    name2feats, name2labels = {}, {}
    for ii in range(len(names)):
        name2feats[names[ii]] = os.path.join(feature_root, names[ii]+'.npy')
        name2labels[names[ii]] = labels[ii]
    return name2feats, name2labels, feature_dim

class MERDataset(Dataset):
    def __init__(self, label_path, audio_root, text_root, video_root, data_type, debug=False):
        assert data_type in ['train', 'test1', 'test2', 'test3']
        self.name2video, self.name2labels, self.vdim = read_data_multiprocess(label_path, video_root, task='whole', data_type=data_type, debug=debug)
        self.name2audio, self.name2labels, self.adim = read_data_multiprocess(label_path, audio_root, task='whole', data_type=data_type, debug=debug)
        self.name2text,  self.name2labels, self.tdim = read_data_multiprocess(label_path, text_root,  task='whole', data_type=data_type, debug=debug)
        # self.name2hog, _, self.hdim = read_data_multiprocess(label_path, hog_root,  task='whole', data_type=data_type, debug=debug)
        # self.name2pose, _, self.pdim = read_data_multiprocess(label_path, pose_root,  task='whole', data_type=data_type, debug=debug)
        self.names = [name for name in self.name2audio if 1==1]
        self.id_to_name = {id:name for id , name in enumerate(self.names)}
        self.name_to_id = {name:id for id , name in enumerate(self.names)}

    def __getitem__(self, index):
        name = self.names[index]
        video_feature = np.load(self.name2video[name]) # load npy files
        audio_feature = np.load(self.name2audio[name]) # load npy files
        text_feature = np.load(self.name2text[name])   # load npy files
        return {
                # "video": torch.FloatTensor(self.name2video[name]),
                # "audio": torch.FloatTensor(self.name2audio[name]),
                # "text":  torch.FloatTensor(self.name2text[name]),
                "video": torch.FloatTensor(video_feature),
                "audio": torch.FloatTensor(audio_feature),
                "text": torch.FloatTensor(text_feature),
                # 'hog': torch.FloatTensor(self.name2hog[name]),
                # 'pose': torch.FloatTensor(self.name2pose[name]),
                "emo": self.name2labels[name]['emo'],
                "val": self.name2labels[name]['val'],
                # "name": name # 'sample_00002547' 
                "name": self.name_to_id[name] # 'sample_00002547' 
                }
    def __len__(self):
        return len(self.names)
    
    def get_id_to_names(self):
        print("Output Id ==> Name e.g. 0 ==> 'sample_00002547' ")
        return self.id_to_name

    def get_featDim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim};')
        return self.adim, self.tdim, self.vdim

###########################################################################################################################################
############################################################ Collate Functions ############################################################
###########################################################################################################################################
def get_key_padding_mask(padded_input, pad_idx):
    """Creates a binary mask to prevent attention to padded locations.
    Arguments
    ----------
    padded_input: int
        Padded input.
    pad_idx:
        idx for padding element.
    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_key_padding_mask(a, pad_idx=0)
    tensor([[False, False,  True],
            [False, False,  True],
            [False, False,  True]])
    """
    if len(padded_input.shape) == 4:
        bz, time, ch1, ch2 = padded_input.shape
        padded_input = padded_input.reshape(bz, time, ch1 * ch2)

    key_padded_mask = padded_input.eq(pad_idx).to(padded_input.device)

    # if the input is more than 2d, mask the locations where they are silence
    # across all channels
    if len(padded_input.shape) > 2:
        key_padded_mask = key_padded_mask.float().prod(dim=-1).bool()
        return key_padded_mask.detach()

    return key_padded_mask.detach()

def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.
    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.
    Returns
    -------
    mask : tensor
        The binary mask.
    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

def get_lookahead_mask(padded_input):
    """Creates a binary mask for each sequence which maskes future frames.
    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input tensor.
    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_lookahead_mask(a)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    seq_len = padded_input.shape[1]
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device))
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach().to(padded_input.device)

def make_masks_transformer( src, tgt, wav_len=None, pad_idx=0):
    """This method generates the masks for training the transformer model.
    Arguments
    ---------
    src : tensor
        The sequence to the encoder (required).
    tgt : tensor
        The sequence to the decoder (required).
    pad_idx : int
        The index for <pad> token (default=0).
    """
    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()

    tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

    src_mask = None
    tgt_mask = get_lookahead_mask(tgt)
    return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

 
def make_masks(src, wav_len=None, pad_idx=0):
    """This method generates the masks for training the transformer model.
    Arguments
    ---------
    src : tensor
        The sequence to the encoder (required).
    pad_idx : int
        The index for <pad> token (default=0).
    """
    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()
    return src_key_padding_mask


def collate_fn(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    ''' 
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: len(x['text']), reverse=True)
    # max_seq_len = max([len(x[0]) for x in batch])

    emo = torch.LongTensor([sample['emo'] for sample in batch])
    val = torch.FloatTensor([sample['val'] for sample in batch])
    name = torch.LongTensor([sample['name'] for sample in batch])
    # name = [sample['name'] for sample in batch]
    
    # audio_data = pad_sequence([torch.FloatTensor(sample['audio']) for sample in batch], batch_first=True)
    # video_data = pad_sequence([torch.FloatTensor(sample['video']) for sample in batch], batch_first=True)
    # text_data  = pad_sequence([torch.FloatTensor(sample['text'])  for sample in batch], batch_first=True)

    audio_data = pad_sequence([torch.FloatTensor(sample['audio']) for sample in batch])
    video_data = pad_sequence([torch.FloatTensor(sample['video']) for sample in batch])
    text_data =  pad_sequence([torch.FloatTensor(sample['text']) for sample in batch])
    # hog_data =  pad_sequence([torch.FloatTensor(sample['hog']) for sample in batch])
    # pose_data = pad_sequence([torch.FloatTensor(sample['pose']) for sample in batch])

    wav_lens = torch.LongTensor([sample['audio'].shape[0] for sample in batch])
    vis_lens = torch.LongTensor([sample['video'].shape[0] for sample in batch])
    txt_lens = torch.LongTensor([sample['text'].shape[0]  for sample in batch])
    # hog_lens = torch.Tensor([sample['hog'].shape[0] for sample in batch])
    # pose_lens = torch.Tensor([sample['pose'].shape[0] for sample in batch])
    
    video_key_padding_mask = ~length_to_mask(vis_lens).bool()  # mask send to video transformer
    audio_key_padding_mask = ~length_to_mask(wav_lens).bool()  # mask send to audio transformer
    text_key_padding_mask  = ~length_to_mask(txt_lens).bool()  # mask send to text transformer
    # hog_key_padding_mask   = ~length_to_mask(hog_lens).bool()  # mask send to video_hog 
    # pose_key_padding_mask  = ~length_to_mask(pose_lens).bool() # mask send to video_pose

    return  video_data, video_key_padding_mask, \
            audio_data, audio_key_padding_mask, \
            text_data,  text_key_padding_mask, \
            emo, val, name


#########################################################################################################################
###################################### for five-fold cross-validation on Train&Val ######################################
#########################################################################################################################
def get_trans_loaders(args, config):
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
                                  collate_fn=collate_fn,
                                  pin_memory=False)
        eval_loader = DataLoader(train_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SubsetRandomSampler(eval_idxs),
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn,
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
                                 collate_fn=collate_fn,
                                 shuffle=False,
                                 pin_memory=False)
        test_loaders.append(test_loader)

    ## return loaders
    adim, tdim, vdim = train_dataset.get_featDim()
    return train_loaders, eval_loaders, test_loaders, adim, tdim, vdim