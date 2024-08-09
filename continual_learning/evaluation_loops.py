# File: evaluation_loops
# ----------------------
# Script containing evaluation loops shared across multiple other scripts

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from time import time
import math
import pdb
from continual_learning.train_utils import accuracy, similarity_block_accuracy, filter_targets, unpack_context_info

def full_eval_epoch(cfg, model, loader, logdir, prefix, eval_mode):
    if eval_mode == "multitask":
        return eval_epoch_multitask(cfg, model, loader, logdir, prefix)
    else:
        return eval_epoch_joint(cfg, model, loader, logdir, prefix)

## Non-JI evaluation ##
def eval_epoch_multitask(cfg, model, dataloader, logdir, prefix):
    dataloader.dataset.eval_scheme = "standard"
    model.eval()
    mode = "multitask"
    total_results = get_std_result_dict(mode)

    model_preds = {}
    criterion = nn.CrossEntropyLoss()
    index_to_token = dataloader.dataset.index_to_token
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            listener_returns = regular_listener_forward_pass(cfg, model, batch, criterion, index_to_token)
            speaker_returns = regular_speaker_rerank_pass(cfg, model, batch, criterion)
            loss = compute_multitask_loss(listener_returns, speaker_returns)

            # Record metrics
            update_multitask_val_metrics(total_results, listener_returns, speaker_returns, batch)
            save_eval_multitask_preds(model_preds, listener_returns, speaker_returns, batch[-1])

    # Save outputs
    savepath = os.path.join(cfg["logdir"], f'{prefix}_standard_preds.pth')
    torch.save(model_preds, savepath)

    process_total_results(total_results, mode)
    return total_results            

def eval_epoch_gen(cfg, model, dataloader, logdir, prefix):
    # Initialize the eval model
    model.eval()

    # Other prelims
    dataloader.dataset.eval_scheme = "generation"
    model_preds = {}
    index_to_token = dataloader.dataset.index_to_token
    with torch.no_grad():
        for curr_iter, (input_tokens, attn_mask, images, image_attn_mask, added_info) in enumerate(tqdm(dataloader)):
            # Perform generation
            device = model.get_listener().device
            images = images.to(device)                
            input_tokens = input_tokens.to(device)
            attn_mask = attn_mask.to(device)
            image_attn_mask = image_attn_mask.to(device)

            # Get the relevant amount of samples
            captions = model.split_generate(
                input_tokens, attn_mask, images, image_attn_mask, dataloader.dataset.processor,
                max_steps=cfg["max_steps"],
                sampling_type=cfg["sampling_type"],
                temperature=cfg["temperature"],
                top_k=cfg["top_k"], top_p=cfg["top_p"],
                repetition_penalty=cfg["repetition_penalty"],
                num_samples=cfg["num_samples"]
            )

            # Save the results
            save_idefics_captions(model_preds, captions, added_info)

    savepath = os.path.join(logdir, f"{prefix}_gen_preds.pth")
    torch.save(model_preds, savepath)

def compute_multitask_loss(listener_returns, speaker_returns):
    l_loss = listener_returns["loss"]
    s_loss = speaker_returns["loss"]
    return (l_loss + s_loss) / 2

def regular_listener_forward_pass(cfg, model, batch, criterion, index_to_token):
    # Perform the forward pass
    input_tokens, attn_mask, images, image_attn_mask, target_label, sim_idx = batch[0]

    # Push to cuda
    device = model.get_listener().device
    input_tokens = input_tokens.to(device)
    attn_mask = attn_mask.to(device)
    images = images.to(device)
    image_attn_mask = image_attn_mask.to(device)
    target_label = target_label.to(device)
    sim_idx = [curr_idx.to(device) for curr_idx in sim_idx]

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        all_logits = model.forward("comprehension",
                                   [
                                       input_tokens,
                                       attn_mask,
                                       images,
                                       image_attn_mask
                                   ])

    target_logits = filter_targets(all_logits[:, -1], index_to_token) # BxC
    loss = criterion(target_logits, target_label)
    
    # Compute the metrics
    acc = accuracy(target_logits, target_label) 
    sim_acc = similarity_block_accuracy(target_logits, sim_idx)

    return {
        "loss" : loss,
        "target_logits" : target_logits,
        "acc" : acc,
        "sim_acc" : sim_acc,
        "B" : images.shape[0]
    }

def regular_speaker_rerank_pass(cfg, model, batch, criterion):
    # Unpack
    input_tokens, attn_mask, images, image_attn_mask, target_tokens, target_mask = batch[1]
    device = model.get_listener().device
    input_tokens = input_tokens.to(device)
    attn_mask = attn_mask.to(device)
    images = images.to(device)
    image_attn_mask = image_attn_mask.to(device)
    target_mask = target_mask.to(device)
    target_tokens = target_tokens.to(device)        

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        utterance_log_probs = model.forward(
            "split_reranking",
            [
                images, input_tokens, attn_mask, image_attn_mask, target_tokens, target_mask
            ]
        )

    # Update the main reranking metrics
    B, mult = utterance_log_probs.shape[:2]
    labels = torch.zeros(B, device=utterance_log_probs.device, dtype=torch.long)
    rerank_acc = accuracy(utterance_log_probs, labels)
    rerank_loss = criterion(utterance_log_probs, labels)

    # Update the soft reranking metrics
    soft_rerank_argmax = torch.sum(utterance_log_probs[:, 0] - torch.argmax(utterance_log_probs[:, 1:], dim=1))

    # Get the number of tokens per utterance
    utterance_lengths = torch.sum(target_mask, dim=-1).view(B, mult)
    alt_split_rerank = accuracy(utterance_log_probs / utterance_lengths, labels)

    loss = - torch.mean(utterance_log_probs[:, 0])
    
    return {
        "loss" : loss,
        "rerank_acc" : rerank_acc,
        "rerank_loss" : rerank_loss,
        "utterance_log_probs" : utterance_log_probs,
        "soft_rerank_argmax" : soft_rerank_argmax,
        "alt_split_rerank_acc" : alt_split_rerank,
    }

## JI evaluation ##
def eval_epoch_joint(cfg, model, dataloader, logdir, prefix):
    dataloader.dataset.eval_scheme = "standard"
    model.eval()
    mode = "joint"
    results = get_std_result_dict(mode)

    model_preds = {}
    criterion = nn.CrossEntropyLoss()
    index_to_token = dataloader.dataset.index_to_token
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # Perform both forward passes
            listener_returns = joint_listener_forward_pass(cfg, model, batch, criterion, index_to_token)
            speaker_returns = joint_speaker_forward_pass(cfg, model, batch, criterion, index_to_token)
            
            # Record metrics
            update_joint_val_metrics(results, listener_returns, speaker_returns, batch)
            save_joint_preds(model_preds, listener_returns, speaker_returns, batch[-1])

    # Save outputs
    savepath = os.path.join(logdir, f'{prefix}_standard_preds.pth')
    torch.save(model_preds, savepath)

    process_total_results(results, mode)
    return results

def eval_epoch_gen_joint(cfg, model, dataloader, logdir, prefix):
    # Initialize the eval model
    model.eval()

    # Other prelims
    dataloader.dataset.eval_scheme = "generation"
    model_preds = {}
    index_to_token = dataloader.dataset.index_to_token
    with torch.no_grad():
        for i, (images, s_input_tokens, s_attn_mask, s_image_attn_mask, label, added_info) in enumerate(tqdm(dataloader)):
            device = model.get_listener().device
            images = images.to(device)                
            label = label.to(device)
            s_input_tokens = s_input_tokens.to(device)
            s_attn_mask = s_attn_mask.to(device)
            s_image_attn_mask = s_image_attn_mask.to(device)

            # Get the relevant amount of samples
            speaker_context = unpack_context_info(added_info["speaker_context"])            
            captions, all_utterances, listener_scores, speaker_scores, joint_scores = model.generate(
                images, s_input_tokens, s_attn_mask, s_image_attn_mask, label,
                speaker_context, dataloader.dataset.processor, dataloader.dataset.img_dir, index_to_token,
                max_steps=cfg["max_steps"],
                sampling_type=cfg["sampling_type"],
                temperature=cfg["temperature"],
                top_k=cfg["top_k"], top_p=cfg["top_p"],
                repetition_penalty=cfg["repetition_penalty"],
                num_samples=cfg["num_samples"]
            )

            # Save the results
            save_idefics_reranked_captions(model_preds, all_utterances, listener_scores, speaker_scores,
                                           joint_scores, added_info, cfg["num_samples"])

    savepath = os.path.join(logdir, f"{prefix}_gen_preds.pth")
    torch.save(model_preds, savepath)

def joint_listener_forward_pass(cfg, model, batch, criterion, index_to_token):
    images, l_input_tokens, l_attn_mask, l_image_attn_mask, s_input_tokens, s_attn_mask, \
        s_image_attn_mask, s_target_mask, s_target_label, label, sim_idx = batch[0]

    device = model.get_listener().device
    images = images.to(device)
    l_input_tokens = l_input_tokens.to(device)
    l_attn_mask = l_attn_mask.to(device)
    l_image_attn_mask = l_image_attn_mask.to(device)
    s_input_tokens = s_input_tokens.to(device)
    s_attn_mask = s_attn_mask.to(device)
    s_image_attn_mask = s_image_attn_mask.to(device)
    s_target_mask = s_target_mask.to(device)
    s_target_label = s_target_label.to(device)
    label = label.to(device)
    sim_idx = [curr_idx.to(device) for curr_idx in sim_idx]

    # Individual forward passes
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        listener_log_probs, speaker_log_probs, joint_log_probs = model.forward(
            "joint_comprehension",
            [
                images, l_input_tokens, l_attn_mask, l_image_attn_mask, index_to_token,
                s_input_tokens, s_attn_mask, s_image_attn_mask, s_target_mask, s_target_label,
            ])

    # Compute the losses
    joint_loss = criterion(joint_log_probs, label)
    split_loss = criterion(listener_log_probs, label)
    loss = (joint_loss + split_loss) / 2            

    # Compute the metrics as well
    split_acc = accuracy(listener_log_probs, label) 
    split_sim_acc = similarity_block_accuracy(listener_log_probs, sim_idx) 
    speaker_acc = accuracy(speaker_log_probs, label)
    speaker_sim_acc = similarity_block_accuracy(speaker_log_probs, sim_idx) 
    joint_acc = accuracy(joint_log_probs, label)
    joint_sim_acc = similarity_block_accuracy(joint_log_probs, sim_idx)

    return  {
        "B" : images.shape[0],
        "listener_log_probs" : listener_log_probs.detach().cpu(),
        "speaker_log_probs" : speaker_log_probs.detach().cpu(),
        "joint_log_probs" : joint_log_probs.detach().cpu(),
        "joint_loss" : joint_loss,
        "split_loss" : split_loss,
        "loss" : loss,
        "split_acc" : split_acc,
        "split_sim_acc" : split_sim_acc,
        "joint_acc" : joint_acc,
        "joint_sim_acc" : joint_sim_acc,
        "speaker_acc" : speaker_acc,
        "speaker_sim_acc" : speaker_sim_acc,
    }

def joint_speaker_forward_pass(cfg, model, batch, criterion, index_to_token):
    images, label, s_input_tokens, s_attn_mask, s_image_attn_mask, s_target_tokens, s_target_mask, \
        l_input_tokens, l_attn_mask, l_image_attn_mask, annotation_mask = batch[1]

    device = model.get_listener().device
    images = images.to(device)
    label = label.to(device)
    l_input_tokens = l_input_tokens.to(device)
    l_attn_mask = l_attn_mask.to(device)
    l_image_attn_mask = l_image_attn_mask.to(device)
    s_input_tokens = s_input_tokens.to(device)
    s_attn_mask = s_attn_mask.to(device)
    s_image_attn_mask = s_image_attn_mask.to(device)
    s_target_mask = s_target_mask.to(device)
    s_target_tokens = s_target_tokens.to(device)
    annotation_mask = annotation_mask.to(device)

    # Get the individual targets for lm forward pass
    main_targets = s_target_tokens[:, 0]
    main_masks = s_target_mask[:, 0]
    B, mult = s_input_tokens.shape[:2]

    # Individual forward passes
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        speaker_logits, speaker_log_probs, listener_log_probs, utterance_distribution = model.forward(
            "joint_reranking",
            [
                images, label, s_input_tokens, s_attn_mask, s_image_attn_mask, s_target_tokens,
                s_target_mask, l_input_tokens, l_attn_mask, l_image_attn_mask, index_to_token,
                annotation_mask
            ])

    # Compute the losses: Joint and split
    B = utterance_distribution.shape[0]
    labels = torch.zeros(B, device=utterance_distribution.device, dtype=torch.long)
    joint_loss = criterion(utterance_distribution, labels)
    split_loss = - torch.mean(speaker_log_probs[:, 0])
    loss = (joint_loss + split_loss) / 2        

    # Compute reranking metrics
    split_rerank = accuracy(speaker_log_probs, labels)
    joint_rerank = accuracy(utterance_distribution, labels)
    listener_rerank = accuracy(listener_log_probs, labels)

    with torch.no_grad():
        soft_rerank_argmax = torch.sum(speaker_log_probs[:, 0] - torch.argmax(speaker_log_probs[:, 1:], dim=1)).item()

        # Get the number of tokens per utterance
        utterance_lengths = torch.sum(s_target_mask, dim=2)
        alt_split_rerank = accuracy(speaker_log_probs / utterance_lengths, labels)

    # Get model predictions for the ground-truths specifically
    _, T, V = speaker_logits.shape
    main_logits = speaker_logits.view(B, mult, T, V)[:, 0]

    return {
        "listener_log_probs" : listener_log_probs.detach().cpu(),
        "speaker_log_probs" : speaker_log_probs.detach().cpu(),
        "joint_log_probs" : utterance_distribution.detach().cpu(),
        "main_log_probs" : main_logits.detach().cpu(),
        "main_targets" : main_targets.detach().cpu(),
        "main_masks" : main_masks.detach().cpu(),
        "joint_loss" : joint_loss,
        "split_loss" : split_loss,
        "loss" : loss,
        "split_rerank_acc" : split_rerank,
        "joint_rerank_acc" : joint_rerank,
        "listener_rerank" : listener_rerank,
        "soft_rerank_argmax" : soft_rerank_argmax,
        "alt_split_rerank_acc" : alt_split_rerank        
    }


## METRIC HELPER FUNCTIONS ##

def get_std_result_dict(eval_mode):
    if eval_mode == "multitask":
        results = {
            "count" : 0, "lm_count" : 0,
            "l" : {
                "loss" : 0, "acc" : 0, "sim_acc" : 0
            },
            "s" : {
                "loss" : 0, "rerank_loss" : 0, "rerank_acc" : 0,
                "soft_rerank_argmax" : 0, "alt_split_rerank_acc" : 0
            }
        }
    else:
        results = {
            "count" : 0, "lm_count" : 0,
            "l" : {
                "split_loss" : 0, "split_acc" : 0, "split_sim_acc" : 0,
                "joint_loss" : 0, "joint_acc" : 0, "joint_sim_acc" : 0,
                "listener_rerank" : 0,
            },
            "s" : {
                "split_loss" : 0, "split_rerank_acc" : 0,
                "joint_loss" : 0, "joint_rerank_acc" : 0,
                "speaker_acc" : 0, "speaker_sim_acc" : 0,
                "soft_rerank_argmax" : 0, "alt_split_rerank_acc" : 0
            }
        }

    return results

def update_joint_val_metrics(results, listener_returns, speaker_returns, batch):
    B = listener_returns["B"]

    for inf in ["joint", "split"]:
        # Update the listener metrics
        results["l"][f'{inf}_loss'] += listener_returns[f"{inf}_loss"].cpu().item() * B
        results["l"][f'{inf}_acc'] += listener_returns[f"{inf}_acc"]
        results["l"][f'{inf}_sim_acc'] += listener_returns[f"{inf}_sim_acc"]
            
        # Update the speaker metrics
        results["s"][f'{inf}_loss'] += speaker_returns[f"{inf}_loss"].cpu().item() * B
        results["s"][f'{inf}_rerank_acc'] += speaker_returns[f"{inf}_rerank_acc"]

    # Update metrics exclusive to split models
    results["l"]["listener_rerank"] += speaker_returns["listener_rerank"]
    results["s"]["soft_rerank_argmax"] += speaker_returns["soft_rerank_argmax"]
    results["s"]["alt_split_rerank_acc"] += speaker_returns["alt_split_rerank_acc"]
    results["s"]["speaker_acc"] += listener_returns["speaker_acc"]
    results["s"]["speaker_sim_acc"] += listener_returns["speaker_sim_acc"]

    results["count"] += B

def update_multitask_val_metrics(results, listener_returns, speaker_returns, batch):
    B = listener_returns["B"]
        
    results["l"]["loss"] += listener_returns["loss"].cpu().item() * B
    results["l"]["acc"] += listener_returns["acc"]
    results["l"]["sim_acc"] += listener_returns["sim_acc"]
    results["s"]["loss"] += speaker_returns["loss"].cpu().item() * B

    results["s"]["rerank_loss"] += speaker_returns["rerank_loss"].cpu().item() * B
    results["s"]["rerank_acc"] += speaker_returns["rerank_acc"]
    results["s"]["soft_rerank_argmax"] += speaker_returns["soft_rerank_argmax"].cpu().item()
    results["s"]["alt_split_rerank_acc"] += speaker_returns["alt_split_rerank_acc"]

    results["count"] += B

def process_total_results(results, mode):
    count = results["count"]
    if mode == "multitask":
        results["l"]["loss"] /= count
        results["l"]["acc"] /= count
        results["l"]["sim_acc"] /= count
        results["s"]["loss"] /= count

        if results["s"]["rerank_loss"] != 0:
            results["s"]["rerank_loss"] /= count
            results["s"]["rerank_acc"] /= count
            results["s"]["soft_rerank_argmax"] /= count
            results["s"]["alt_split_rerank_acc"] /= count                    
    else:
        for inf in ["split", "joint"]:
            for postfix in ["loss", "acc", "sim_acc"]:
                results["l"][f"{inf}_{postfix}"] /=  count
            for postfix in ["loss", "rerank_acc"]:
                results["s"][f"{inf}_{postfix}"] /= count

        # Split only results
        results["l"]["listener_rerank"] /= count
        results["s"]["speaker_acc"] /= count
        results["s"]["speaker_sim_acc"] /= count
        results["s"]["soft_rerank_argmax"] /= count
        results["s"]["alt_split_rerank_acc"] /= count        

def save_eval_multitask_preds(model_preds, listener_returns, speaker_returns, added_info):
    model_output = listener_returns["target_logits"].detach().cpu()
    for i in range(model_output.shape[0]):    
        game_id = added_info["game_id"][i]
        round_idx = added_info["round_index"][i]
        listener_context = [added_info["listener_context"][j][i] for j in range(10)]
        speaker_context = [added_info["speaker_context"][j][i] for j in range(10)]

        if game_id not in model_preds:
            model_preds[game_id] = {}
        model_preds[game_id][round_idx] = {
            "listener_context" : listener_context,
            "speaker_context" : speaker_context,
            "listener_output" : model_output[i]
        }

        utterance_log_probs = speaker_returns["utterance_log_probs"][i].detach().cpu()
        model_preds[game_id][round_idx]["utterance_log_probs"] = utterance_log_probs

def save_joint_preds(model_preds, listener_returns, speaker_returns, added_info):
    speaker_logits = speaker_returns["main_log_probs"]
    speaker_targets = speaker_returns["main_targets"]
    speaker_masks = speaker_returns["main_masks"]

    for i in range(len(added_info["game_id"])):
        game_id = added_info["game_id"][i]
        round_idx = added_info["round_index"][i]
        listener_context = [added_info["listener_context"][j][i] for j in range(10)]
        speaker_context = [added_info["speaker_context"][j][i] for j in range(10)]

        # Get the filtered speaker preds
        speaker_logit = speaker_logits[i][speaker_masks[i]]
        speaker_target = speaker_targets[i][speaker_masks[i]]

        if game_id not in model_preds:
            model_preds[game_id] = {}
        model_preds[game_id][round_idx] = {
            "listener_context" : listener_context,
            "speaker_context" : speaker_context,
            "l_listener_probs" : listener_returns["listener_log_probs"][i],
            "l_speaker_probs" : listener_returns["speaker_log_probs"][i],
            "l_joint_probs" : listener_returns["joint_log_probs"][i],
            "s_listener_probs" : speaker_returns["listener_log_probs"][i],
            "s_speaker_probs" : speaker_returns["speaker_log_probs"][i],
            "s_joint_probs" : speaker_returns["joint_log_probs"][i],
            "main_s_logits" : speaker_logit,
            "main_s_target" : speaker_target,
        }

def save_idefics_captions(model_preds, captions, added_info):
    for i in range(len(added_info["game_id"])):
        game_id = added_info["game_id"][i]
        round_idx = added_info["round_index"][i]
        example_context = [added_info["context"][j][i] for j in range(10)]                

        if game_id not in model_preds:
            model_preds[game_id] = {}
        model_preds[game_id][round_idx] = {
            "context" : example_context,
            "utterances" : captions[i],
        }

def save_idefics_reranked_captions(model_preds, all_utterances, listener_scores, speaker_scores,
                                   joint_scores, added_info, num_samples):
    for i in range(len(added_info["game_id"])):
        game_id = added_info["game_id"][i]
        round_idx = added_info["round_index"][i]
        listener_context = [added_info["listener_context"][j][i] for j in range(10)]
        speaker_context = [added_info["speaker_context"][j][i] for j in range(10)]                
        if game_id not in model_preds:
            model_preds[game_id] = {}

        model_preds[game_id][round_idx] = {
            "listener_context" : listener_context,
            "speaker_context" : speaker_context,
            "utterances" : all_utterances[i*num_samples:(i+1)*num_samples],
            "listener_scores" : listener_scores[i].detach().cpu(),
            "speaker_scores" : speaker_scores[i].detach().cpu(),
            "joint_scores" : joint_scores[i].detach().cpu(),
        }


