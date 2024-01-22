#! python3
# -*- encoding: utf-8 -*-
#####################################################################################################################################################################################
############################################################################## metric and save results ##############################################################################
#####################################################################################################################################################################################
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

emos = ['neutral', 'angry', 'happy', 'sad', 'worried',  'surprise']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos): emo2idx[emo] = ii
for ii, emo in enumerate(emos): idx2emo[ii] = emo


def overall_metric(emo_fscore, val_mse):
    final_score = emo_fscore - val_mse * 0.25
    return final_score


def average_folder_results(folder_save, testname):
    name2preds = {}
    num_folder = len(folder_save)
    for ii in range(num_folder):
        names    = folder_save[ii][f'{testname}_names']
        emoprobs = folder_save[ii][f'{testname}_emoprobs']
        valpreds = folder_save[ii][f'{testname}_valpreds']
        for jj in range(len(names)):
            name = names[jj]
            emoprob = emoprobs[jj]
            valpred = valpreds[jj]
            if name not in name2preds: name2preds[name] = []
            name2preds[name].append({'emo': emoprob, 'val': valpred})

    ## gain average results
    name2avgpreds = {}
    for name in name2preds:
        preds = np.array(name2preds[name])
        emoprobs = [pred['emo'] for pred in preds if 1==1]
        valpreds = [pred['val'] for pred in preds if 1==1]

        avg_emoprob = np.mean(emoprobs, axis=0)
        avg_emopred = np.argmax(avg_emoprob)
        avg_valpred = np.mean(valpreds)
        name2avgpreds[name] = {'emo': avg_emopred, 'val': avg_valpred, 'emoprob': avg_emoprob}
    return name2avgpreds

def gain_name2feat(folder_save, testname):
    name2feat = {}
    assert len(folder_save) >= 1
    names      = folder_save[0][f'{testname}_names']
    embeddings = folder_save[0][f'{testname}_embeddings']
    for jj in range(len(names)):
        name = names[jj]
        embedding = embeddings[jj]
        name2feat[name] = embedding
    return name2feat

def write_to_csv_pred(name2preds, save_path):
    names, emos, vals = [], [], []
    for name in name2preds:
        names.append(name)
        emos.append(idx2emo[name2preds[name]['emo']])
        vals.append(name2preds[name]['val'])

    columns = ['name', 'discrete', 'valence']
    data = np.column_stack([names, emos, vals])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)

def report_results_on_test1_test2(test_label, test_pred):
    # read target file (few for test3)
    name2label = {}
    df_label = pd.read_csv(test_label)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        val  = row['valence']
        name2label[name] = {'emo': emo2idx[emo], 'val': val}
    print (f'labeled samples: {len(name2label)}')

    # read prediction file (more for test3)
    name2pred = {}
    df_label = pd.read_csv(test_pred)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        val  = row['valence']
        name2pred[name] = {'emo': emo2idx[emo], 'val': val}
    print (f'predict samples: {len(name2pred)}')
    assert len(name2pred) == len(name2label), f'make sure len(name2pred)=len(name2label)'

    emo_labels, emo_preds, val_labels, val_preds = [], [], [], []
    for name in name2label:
        emo_labels.append(name2label[name]['emo'])
        val_labels.append(name2label[name]['val'])
        emo_preds.append(name2pred[name]['emo'])
        val_preds.append(name2pred[name]['val'])

    # analyze results
    emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')
    print (f'emo results (weighted f1 score): {emo_fscore:.4f}')
    val_mse = mean_squared_error(val_labels, val_preds)
    print (f'val results (mse): {val_mse:.4f}')
    final_metric = overall_metric(emo_fscore, val_mse)
    print (f'overall metric: {final_metric:.4f}')
    return emo_fscore, val_mse, final_metric


## only fscore for test3
def report_results_on_test3(test_label, test_pred):

    # read target file (few for test3)
    name2label = {}
    df_label = pd.read_csv(test_label)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        name2label[name] = {'emo': emo2idx[emo]}
    print (f'labeled samples: {len(name2label)}')

    # read prediction file (more for test3)
    name2pred = {}
    df_label = pd.read_csv(test_pred)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        name2pred[name] = {'emo': emo2idx[emo]}
    print (f'predict samples: {len(name2pred)}')
    assert len(name2pred) >= len(name2label)

    emo_labels, emo_preds = [], []
    for name in name2label: # on few for test3
        emo_labels.append(name2label[name]['emo'])
        emo_preds.append(name2pred[name]['emo'])

    # analyze results
    emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')
    print (f'emo results (weighted f1 score): {emo_fscore:.4f}')
    return emo_fscore, -100, -100