import os
import time
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
import torch.optim as optim
from prettytable import PrettyTable
from torch.utils.data import DataLoader, SubsetRandomSampler

# Import custom modules and classes
from utils import overall_metric
from losses.loss import MSELoss, CELoss
from datasets import get_trans_loaders
from engines.engine_trans_logits import train_or_eval_model
from models import MLP, EfficientMultimodalTransformer, MultimmodalCATransformer, MultimodalBottleneckTransformer, UnifiedMultimodalTransformer
import config

def main(args, configs):
    # Initialize arguments
    args.n_classes = 6
    args.num_folder = 5
    args.test_sets = args.test_sets.split(',')
    
    if args.dataset is not None:
        args.train_dataset = args.dataset
        args.test_dataset  = args.dataset
    assert args.train_dataset is not None
    assert args.test_dataset  is not None    
    
    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    feature_count = len(set(whole_features))
    
    if feature_count == 1:
        args.save_root = f'{args.save_root}-unimodal'
    elif feature_count == 2:
        args.save_root = f'{args.save_root}-bimodal'
    elif feature_count == 3:
        args.save_root = f'{args.save_root}-trimodal'

    torch.cuda.set_device(args.gpu)
    print(args)

    print(f'====== Reading Data =======')
    train_loaders, eval_loaders, test_loaders, adim, tdim, vdim = get_trans_loaders(args, config)

    assert len(train_loaders) == args.num_folder, 'Error: folder number'
    assert len(eval_loaders)   == args.num_folder, 'Error: folder number'
    
    print(f'====== Training and Evaluation =======')    
    for fold in range(args.num_folder):
        print(f'>>>>> Cross-validation: training on the {fold+1} folder >>>>>')
        train_loader = train_loaders[fold]
        eval_loader  = eval_loaders[fold]
        start_time = time.time()
        name_time  = time.time()
        
        print(f'Step1: Build Model (Each folder has its own model)')
        if args.model_type == 'mlp':
            model = MLP(input_dim=adim + tdim + vdim,
                        output_dim1=args.n_classes,
                        output_dim2=1,
                        layers=args.layers)
        elif args.model_type == 'emt':
            model = EfficientMultimodalTransformer(
                              in_chans=1024,
                              d_model=768)
        elif args.model_type == 'mbt':
            model = MultimodalBottleneckTransformer(
                              in_chans=1024,
                              d_model=768)
        elif args.model_type == 'umt':
            model = UnifiedMultimodalTransformer(
                              in_chans=1024,
                              d_model=768)
        elif args.model_type == 'ca':
            model = MultimmodalCATransformer(
                            in_chans=1024,
                            d_model=768)
        else:
            raise NotImplementedError
        
        weights = torch.tensor([1.0, 1.0, 1.0]).cuda()
        weights = torch.nn.Parameter(weights)
        reg_loss = MSELoss()
        cls_loss = CELoss()
        
        model.cuda()
        reg_loss.cuda()
        cls_loss.cuda()
        
        optimizer_logits = optim.Adam([weights], lr=args.lr, weight_decay=args.l2)        
        optimizer_network = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        print(f'Step2: Training (Multiple Epochs)')
        
        # Start training
        iteration = 0 
        for epoch in range(args.epochs):
            # Training and validation            
            train_results = train_or_eval_model(args, model, cls_loss, train_loader, optimizer_network=optimizer_network,
                                                optimizer_logits=optimizer_logits, weights=weights, iteration=iteration, train=True)
            eval_results = train_or_eval_model(args, model, cls_loss, eval_loader, optimizer_network=None, optimizer_logits=None, train=False)
            iteration += 1
            
            # Print classification per epoch
            if epoch % 5 == 0:                
                # Using PrettyTable
                pt = PrettyTable()
                pt.field_names = ["Epoch", "Vision Performance", "Audio Performance", "Fusion Performance"]
                pt.add_row([epoch+1, eval_results["emo_report_v"],eval_results["emo_report_a"],eval_results["emo_report_b"]])
                print(pt)

    end_time = time.time()
    print(f'>>>>> Finish: training on the {fold+1} folder, duration: {end_time - start_time} >>>>>')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## params for input
    parser.add_argument('--config', type=str, default='./configs/train_val_test.yaml', help='config files')
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--dataset', type=str, default=None, help='dataset, CREMAD and AVE')
    parser.add_argument('--fps', type=int, default=1, help='frame per second')
    parser.add_argument('--fusion_method', type=str, default='concat', help='fusion method: sum, concat, film, gated')
    parser.add_argument('--train_dataset', type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--test_dataset',  type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--audio_feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text_feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video_feature', type=str, default=None, help='video feature name')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--test_sets', type=str, default='test3', help='process on which test sets, [test1, test2, test3]')
    parser.add_argument('--save_root', type=str, default='./saved', help='save prediction results and models')
    parser.add_argument('--savewhole', action='store_true', default=False, help='whether save latent embeddings')

    ## params for model
    parser.add_argument('--layers', type=str, default='256,128', help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=-1, help='number of classes [defined by args.label_path]')
    parser.add_argument('--num_folder', type=int, default=-1, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type for training [mlp or attention]')

    ## params for training
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0002, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
        
    args = parser.parse_args()
    configs = yaml.load(open(args.config,'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(configs, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))        
    main(args, configs)