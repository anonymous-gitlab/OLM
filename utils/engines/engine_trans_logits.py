import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report

def train_or_eval_model(args, model, cls_loss, dataloader, optimizer_network=None, optimizer_logits=None, weights=None, iteration=0, train=False):
    '''
    Train or evaluate the model.

    Args:
        args (dict): Configuration arguments.
        model (nn.Module): The neural network model.
        cls_loss (callable): Loss function for classification.
        dataloader (DataLoader): Training or evaluation data loader.
        optimizer_network (Optimizer, optional): Optimizer for the entire network.
        optimizer_logits (Optimizer, optional): Optimizer for the logit modulation module.
        weights (list, optional): Coefficients for each modal logit vector.
        iteration (int, optional): Flag for recording the initial loss of the first training epoch.
        train (bool, optional): Train mode if True, evaluate mode if False.

    Returns:
        dict: Results including accuracy, classification reports, logits, and video names.
    '''

    vidnames = []
    emo_logits_v, emo_logits_a, emo_logits_b, emo_labels = [], [], [], []

    assert not train or optimizer_network is not None
    model.train() if train else model.eval()

    for data in dataloader:
        visual_feat, visual_mask, audio_feat, audio_mask, text_feat, text_mask = data[:6]
        emos, vals = data[6], data[7].float()
        vidnames += data[-1]

        # Move data to CUDA if available
        emos, vals, audio_feat, audio_mask, text_feat, text_mask, visual_feat, visual_mask = map(lambda x: x.cuda(), 
                                                                                                  [emos, vals, audio_feat, audio_mask, text_feat, text_mask, visual_feat, visual_mask])

        
        emos_out_v, emos_out_a, emos_out_b = model(visual_feat, visual_mask, audio_feat, audio_mask, text_feat, text_mask)
        
        # Collect logits and labels
        emo_logits_v.append(emos_out_v.data.cpu().numpy())
        emo_logits_a.append(emos_out_a.data.cpu().numpy())
        emo_logits_b.append(emos_out_b.data.cpu().numpy())
        emo_labels.append(emos.data.cpu().numpy())

        # Optimize parameters during training
        if train:
            logits = torch.stack([emos_out_v, emos_out_a, emos_out_b])
            loss_v = cls_loss(weights[0] * emos_out_v, emos)
            loss_a = cls_loss(weights[1] * emos_out_a, emos)
            loss_b = cls_loss(weights[2] * emos_out_b, emos)
            loss = loss_v + loss_a + loss_b

            if iteration == 0:
                model.initial_loss = torch.stack([loss_v, loss_a, loss_b]).detach()

            optimizer_network.zero_grad()
            optimizer_logits.zero_grad()
            loss.backward()

            logits_norm = [weights[i] * torch.norm(logit, dim=-1).detach() for i, logit in enumerate(logits)]

            # Optimize logit coefficients
            logits_norm = torch.stack(logits_norm, dim=-1)
            loss_ratio = torch.stack([loss_v, loss_a, loss_b]).detach() / model.initial_loss
            rt = loss_ratio / loss_ratio.mean()
            logits_norm_avg = logits_norm.mean(-1).detach()
            constant = (logits_norm_avg.unsqueeze(-1) @ rt.unsqueeze(0)).detach()
            logitsnorm_loss = torch.abs(logits_norm - constant).sum()
            logitsnorm_loss.backward()

            optimizer_network.step()
            optimizer_logits.step()

    # Evaluate on discrete labels
    emo_labels = np.concatenate(emo_labels)
    emo_logits_v = np.concatenate(emo_logits_v)
    emo_logits_a = np.concatenate(emo_logits_a)
    emo_logits_b = np.concatenate(emo_logits_b)
    emo_preds_v = np.argmax(emo_logits_v, 1)
    emo_preds_a = np.argmax(emo_logits_a, 1)
    emo_preds_b = np.argmax(emo_logits_b, 1)

    emo_acc_v = accuracy_score(emo_labels, emo_preds_v)
    emo_acc_a = accuracy_score(emo_labels, emo_preds_a)
    emo_acc_b = accuracy_score(emo_labels, emo_preds_b)

    emo_report_v = classification_report(emo_labels, emo_preds_v, zero_division=1)
    emo_report_a = classification_report(emo_labels, emo_preds_a, zero_division=1)
    emo_report_b = classification_report(emo_labels, emo_preds_b, zero_division=1)

    save_results = {
        'emo_labels': emo_labels,
        'emo_acc_v': emo_acc_v,
        'emo_acc_a': emo_acc_a,
        'emo_acc_b': emo_acc_b,
        'emo_report_v': emo_report_v,
        'emo_report_a': emo_report_a,
        'emo_report_b': emo_report_b,
        'emo_logits_v': emo_logits_v,
        'emo_logits_a': emo_logits_a,
        'emo_logits_b': emo_logits_b,
        'names': vidnames
    }

    return save_results