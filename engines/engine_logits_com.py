import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report

def train_or_eval_model(args, 
                        model, 
                        cls_loss, 
                        train_dataloader, 
                        eval_dataloader,
                        optimizer_network=None, 
                        optimizer_logits=None, 
                        weights=None, 
                        iteration=0, 
                        train=False):
    '''
    Train or evaluate the model.

    Args:
        args (dict): Configuration arguments.
        model (nn.Module): The neural network model.
        cls_loss (callable): Loss function for classification.
        train_dataloader (DataLoader): Training or evaluation data loader.
        eval_dataloader (DataLoader): Training or evaluation data loader.
        optimizer_network (Optimizer, optional): Optimizer for the entire network.
        optimizer_logits (Optimizer, optional): Optimizer for the logit modulation module.
        weights (list, optional): Coefficients for each modal logit vector.
        iteration (int, optional): Flag for recording the initial loss of the first training epoch.
        train (bool, optional): Train mode if True, evaluate mode if False.

    Returns:
        dict: Results including accuracy, classification reports, logits, and video names.
    '''
    
    emo_logits_v, emo_logits_a, emo_logits_b, emo_labels = [], [], [], []
    assert not train or optimizer_network is not None
    model.train() if train else model.eval()

    for train_data, eval_data in zip(train_dataloader,eval_dataloader):
        t_spectrogram, t_images, t_label = train_data[:3]
        train_specs, train_imgs = t_spectrogram.unsqueeze(1).float(), t_images.float()
        
        e_spectrogram, e_images, e_label = eval_data[:3]
        eval_specs, eval_imgs = e_spectrogram.unsqueeze(1).float(), e_images.float()
        
        # Move data to CUDA if available
        train_audio, train_image, train_emos = map(lambda x: x.cuda(),[train_specs, train_imgs, t_label])
        eval_audio, eval_image, eval_emos = map(lambda x: x.cuda(),[eval_specs, eval_imgs, e_label])
        
        emos_out_a, emos_out_v, emos_out_b = model(train_audio, train_image, train_emos, train)
        
        # Collect logits and labels
        emo_logits_v.append(emos_out_v.data.cpu().numpy())
        emo_logits_a.append(emos_out_a.data.cpu().numpy())
        emo_logits_b.append(emos_out_b.data.cpu().numpy())
        emo_labels.append(t_label.data.cpu().numpy())
        
        # Optimize parameters during training
        if train:
            emos_out_a_eval, emos_out_v_eval, emos_out_b_eval = model(eval_audio, eval_image, eval_emos, train)
            logits = torch.stack([emos_out_v, emos_out_a, emos_out_b])
            loss_v = cls_loss(weights[0] * emos_out_v, train_emos)
            loss_a = cls_loss(weights[1] * emos_out_a, train_emos)
            loss_b = cls_loss(weights[2] * emos_out_b, train_emos)
            loss_v_eval = cls_loss(weights[0] * emos_out_v_eval, eval_emos)
            loss_a_eval = cls_loss(weights[1] * emos_out_a_eval, eval_emos)
            loss_b_eval = cls_loss(weights[2] * emos_out_b_eval, eval_emos)
            loss = loss_v + loss_a + loss_b

            if iteration == 0:
                model.train_initial_loss = torch.stack([loss_v, loss_a, loss_b]).detach()
                model.eval_initial_loss = torch.stack([loss_v_eval, loss_a_eval, loss_b_eval]).detach()

            optimizer_network.zero_grad()
            optimizer_logits.zero_grad()
            loss.backward()

            logits_norm = [weights[i] * torch.norm(logit, dim=-1).detach() for i, logit in enumerate(logits)]

            # Optimize logit coefficients
            logits_norm = torch.stack(logits_norm, dim=-1)
            generalization_rate = model.eval_initial_loss - torch.stack([loss_v_eval, loss_a_eval, loss_b_eval]).detach()
            convergence_rate = model.train_initial_loss - torch.stack([loss_v, loss_a, loss_b]).detach() - generalization_rate
            rt = torch.softmax(torch.clamp(generalization_rate, 1e-6) / torch.clamp(convergence_rate, 1e-6),dim=-1)
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
    }

    return save_results