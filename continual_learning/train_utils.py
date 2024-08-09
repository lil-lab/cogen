# File: train_utils
# -----------------
# Contains basic functions shared across multiple training and
# evaluation scripts

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#### TRAINING TIME ####

def filter_targets(masked_logits, index_to_token):
    target_logits = masked_logits[:, index_to_token]
    return target_logits

def get_target_logits(all_logits, index_to_token):
    # Input: BxTxV
    final_logits = all_logits[:, -1]
    target_logits = filter_targets(final_logits, index_to_token)
    return target_logits

def unpack_context_info(context):
    new_context = []
    for i in range(len(context[0])):
        curr_context = [context[j][i] for j in range(10)]
        new_context.append(curr_context)
    return new_context

def get_next_batch(load_dict, pre):
    if load_dict[f"{pre}_steps"] >= len(load_dict[f"{pre}_loader"]):
        load_dict[f"{pre}_iter"] = iter(load_dict[f"{pre}_loader"])
        load_dict[f"{pre}_steps"] = 0
    return_batch = next(load_dict[f"{pre}_iter"])
    load_dict[f"{pre}_steps"] += 1
    return return_batch

def get_sub_losses(losses, rewards):
    with torch.no_grad():
        pos_mask = rewards == 1
        pos_count = torch.sum(pos_mask).item()
        if pos_count > 0:
            pos_loss = torch.sum(losses[pos_mask]).item()
        else:
            pos_loss = 0

        neg_mask = rewards != 1
        neg_count = torch.sum(neg_mask).item()
        if neg_count > 0:
            neg_loss = torch.sum(losses[neg_mask]).item()
        else:
            neg_loss = 0

    return pos_loss, neg_loss, pos_count, neg_count

#### METRICS ####
def accuracy(model_outputs, labels):
    '''
    Inputs:
      * model_outputs: Where the index with the maximum value is the prediction.
                       Shape BxC
      * labels:        Contains the indices of the target classes. Shape B.
    '''
    with torch.no_grad():
        model_preds = torch.argmax(model_outputs, dim=1)
        matches = (model_preds == labels).float()
        acc = torch.sum(matches).item()
    return acc    

def cl_accuracies(target_logits, reward, gt_label, target_label):
    pos_count, pos_acc = slice_accuracy(target_logits, reward, 1, gt_label)
    neg_count, neg_acc = slice_accuracy(target_logits, reward, -1, gt_label)
    neg_mismatch = slice_mismatch(target_logits, reward, -1, target_label)
    return pos_count, pos_acc, neg_count, neg_acc, neg_mismatch

def slice_accuracy(logits, rewards, reward_sign, gt_labels):
    with torch.no_grad():
        reward_mask = rewards == reward_sign
        if torch.sum(reward_mask).item() == 0:
            return 0, 0
        else:
            logit_slice = logits[reward_mask]
            label_slice = gt_labels[reward_mask]
            count = logit_slice.shape[0]
            acc = accuracy(logit_slice, label_slice)
            return count, acc

def slice_mismatch(logits, rewards, reward_sign, target_labels):
    with torch.no_grad():
        reward_mask = rewards == reward_sign
        if torch.sum(reward_mask.float()).item() == 0:
            return 0
        else:
            logit_slice = logits[reward_mask]
            label_slice = target_labels[reward_mask]
            mismatch = mismatches(logit_slice, label_slice)
            return mismatch

def mismatches(model_outputs, labels):
    with torch.no_grad():
        model_preds = torch.argmax(model_outputs, dim=1)
        matches = (model_preds != labels).float()
        mismatch = torch.sum(matches).item()
    return mismatch    
    
def similarity_block_accuracy(model_outputs, block_indices):
    '''
    Inputs:
      * model_outputs: Where the index with the maximum value is the prediction.
                       Shape BxC
      * block_indices: A list of lists containing the indices of similarity blocks
    '''
    corrects = 0
    with torch.no_grad():
        model_preds = torch.argmax(model_outputs, dim=1)
        for i in range(len(model_preds)):
            model_pred = model_preds[i].item()
            curr_indices = [block_indices[j][i].item() for j in range(4)]
            if model_pred in curr_indices:
                corrects += 1
    return corrects
