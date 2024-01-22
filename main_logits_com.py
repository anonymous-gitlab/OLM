import os
import time
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
import torch.optim as optim
from prettytable import PrettyTable
from models import MLP, AttentionLogits, ConcatLogits, SummationLogits, AVClassifier
from engines.engine_logits_com import train_or_eval_model
from losses.loss import MSELoss, CELoss


from datasets import (
    get_ave_loaders,
    get_cremad_loaders,
    get_vggsound_loaders,
    get_vggsound_loaders,
    get_ks_loaders
)

def main(args, configs):
    # Initialize configuration
    initialize_config(args)

    # Get data loaders based on the selected dataset
    train_loaders, eval_loaders, test_loaders, adim, tdim, vdim = get_data_loaders(args, configs)

    print(f'====== Training and Evaluation =======')

    for fold in range(args.num_folder):
        print(f'>>>>> Cross-validation: training on folder {fold + 1} >>>>>')
        train_loader = train_loaders[fold]
        eval_loader = eval_loaders[fold]
        start_time = time.time()

        # Build and initialize model
        model, weights, reg_loss, cls_loss, optimizer_network, optimizer_logits = initialize_model(args, adim, tdim, vdim)

        print(f'Step 2: training (multiple epochs)')

        iteration = 0
        for epoch in range(args.epochs):
            train_results = train_or_eval_model(args, model, cls_loss, train_loader, eval_loader, optimizer_network, optimizer_logits, weights, iteration, train=True)
            eval_results = train_or_eval_model(args, model, cls_loss, train_loader, eval_loader)
            iteration += 1

            # Print classification per epoch
            if epoch % 5 == 0:
                print_classification_results(epoch, eval_results)

    end_time = time.time()
    print(f'>>>>> Finish: training on folder {fold + 1}, duration: {end_time - start_time} >>>>>')

def initialize_config(args):
    args.num_folder = 5
    args.test_sets = args.test_sets.split(',')

    if args.dataset is not None:
        args.train_dataset = args.dataset
        args.test_dataset = args.dataset
    assert args.train_dataset is not None
    assert args.test_dataset is not None

    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    feature_count = len(set(whole_features))
    args.save_root += f'-{feature_count}modal'

    torch.cuda.set_device(args.gpu)
    print(args)

def get_data_loaders(args, configs):
    dataset_name = args.dataset.lower()
    if dataset_name == 'ave':
        train_loaders, eval_loaders, test_loaders, adim, tdim, vdim = get_ave_loaders(args, configs)
    elif dataset_name == 'cremad':
        train_loaders, eval_loaders, test_loaders, adim, tdim, vdim = get_cremad_loaders(args, configs)
    elif dataset_name == 'vggsound':
        train_loaders, eval_loaders, test_loaders, adim, tdim, vdim = get_vggsound_loaders(args, configs)
    elif dataset_name == 'kineticsound':
        train_loaders, eval_loaders, test_loaders, adim, tdim, vdim = get_ks_loaders(args, configs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    assert len(train_loaders) == args.num_folder, f'Error: folder number'
    assert len(eval_loaders) == args.num_folder, f'Error: folder number'

    return train_loaders, eval_loaders, test_loaders, adim, tdim, vdim

def initialize_model(args, adim, tdim, vdim):
    # Initialize the model based on the selected model_type
    if args.model_type == 'avc':
        model = AVClassifier(args)

    weights = torch.tensor([1.0, 1.0, 1.0]).cuda()
    weights = torch.nn.Parameter(weights)

    reg_loss = MSELoss()
    cls_loss = CELoss()

    model.cuda()
    reg_loss.cuda()
    cls_loss.cuda()

    optimizer_logits = optim.Adam([weights], lr=0.001)
    optimizer_network = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    return model, weights, reg_loss, cls_loss, optimizer_network, optimizer_logits

def print_classification_results(epoch, eval_results):
    pt = PrettyTable()
    pt.field_names = ["Epoch", "Vision Performance", "Audio Performance", "Fusion Performance"]
    pt.add_row([epoch + 1, eval_results["emo_report_v"], eval_results["emo_report_a"], eval_results["emo_report_b"]])
    print(pt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # params for input
    parser.add_argument('--config', type=str, default='./configs/train_val_test.yaml', help='config files')
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--dataset', type=str, default='AVE', help='dataset, CREMAD, AVE, VGGSound, KS')
    parser.add_argument('--fps', type=int, default=1, help='frame per second')
    parser.add_argument('--fusion_method', type=str, default='concat', help='fusion method: sum, concat, film, gated')
    parser.add_argument('--train_dataset', type=str, default=None, help='dataset')  # for cross-dataset evaluation
    parser.add_argument('--test_dataset', type=str, default=None, help='dataset')  # for cross-dataset evaluation
    parser.add_argument('--audio_feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text_feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video_feature', type=str, default=None, help='video feature name')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--test_sets', type=str, default='test1,test2', help='process on which test sets, [test1, test2, test3]')
    parser.add_argument('--save_root', type=str, default='./saved', help='save prediction results and models')
    parser.add_argument('--savewhole', action='store_true', default=False, help='whether save latent embeddings')

    # params for model
    parser.add_argument('--layers', type=str, default='256,128', help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=-1, help='number of classes [defined by args.label_path]')
    parser.add_argument('--num_folder', type=int, default=-1, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--model_type', type=str, default='avc', help='model type for training [avc]')

    # params for training
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0002, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')

    args = parser.parse_args()
    configs = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(configs, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, configs)