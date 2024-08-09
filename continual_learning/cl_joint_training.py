# File: joint_training
# --------------------
# Main script for continual learning training

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import os
from tqdm import tqdm
from time import time
import math

from continual_learning.train_utils import cl_accuracies, filter_targets, get_next_batch, get_sub_losses
from continual_learning.evaluation_loops import full_eval_epoch

from models.model_utils import get_cl_idefics_joint_model, save_joint_idefics_checkpoint
from models.optim_utils import get_idefics_optimizer_and_scheduler
from data_utils.dataset import get_idefics_loader, get_cl_idefics_loaders

from utils.utils import setup_cl_experiment, construct_config, setup_wandb
from utils.utils import add_cl_experiment_arguments, add_training_arguments

def get_config():
    parser = argparse.ArgumentParser()

    # Key arguments
    parser.add_argument('--shared_parameters', action='store_true',
                        help="If set, we will have the comprehension and generation models share parameters")
    parser.add_argument('--training_type', type=str,
                        help="Whether to perform multi-task learning or experiment with alternative objectives")
    parser.add_argument('--evaluation_type', type=str,
                        help="Whether to perform joint evaluation or just multi-task evaluation")
    parser.add_argument('--listener_lambda', type=float,
                        help="The lambda value for listener inference")
    parser.add_argument('--speaker_lambda', type=float,
                        help="The lambda value for speaker inference")

    # Optimization arguments
    parser.add_argument('--learning_rate', type=float,
                        help="Learning rate for the generation model")
    parser.add_argument('--weight_decay', type=float, 
                        help="Weight decay for the comprehension model")
    parser.add_argument('--num_training_steps', type=int,
                        help="The maximum number of steps to train a model for")
    parser.add_argument('--num_warmup_steps', type=int,
                        help="The number of gradient steps to warmup to the maximum lr")
    parser.add_argument('--ips_clip', type=float,
                        help="Whether to clip the IPS term for added stability in first epoch")
    parser.add_argument('--gradient_accumulation_steps', type=int,
                        help="Number of gradient accumulation steps to perform")
    parser.add_argument('--gradient_clip_norm', type=float,
                        help="The amount of gradient clipping to apply")

    # Model hyperparameters
    parser.add_argument('--generation_prompt', type=str,
                        help="The prompt type to use in generation")
    parser.add_argument('--comprehension_prompt', type=str,
                        help="The prompt type to use in generation")
    parser.add_argument('--lora_r', type=int,
                        help="The low-rank dimension to use for LoRA")

    # Training and experiment arguments
    add_training_arguments(parser)
    add_cl_experiment_arguments(parser)

    args = parser.parse_args()
    config = construct_config(args, "idefics_cl_joint_training.yaml")

    return config


def train(cfg, model, optimizer, scheduler, tr_loaders, val_loader, mode, eval_mode):
    # Training setup
    best_metrics = get_best_metrics(cfg, eval_mode)
    epochs_since_improvement = cfg["epochs_since_improvement"]
    start_time = time()

    treatment = cfg["model_family_name"]
    split_dataloaders = cfg["use_separate_dataloaders"] or cfg["model_family_name"] in ["no_ds", "baseline"]
    main_loader_name = "main_listener_loader" if split_dataloaders else "main_loader"
    epoch_size = len(tr_loaders[main_loader_name])
    for epoch in range(cfg["start_epoch"], cfg["n_epochs"]):
        # Clear cache
        torch.cuda.empty_cache()

        # Train epoch
        tr_metrics, is_nan = train_epoch(cfg, epoch, model, optimizer, scheduler, tr_loaders, start_time, mode)
        print_tr_results(tr_metrics)

        # Validation loop
        val_metrics = full_eval_epoch(cfg, model, val_loader, cfg["logdir"], f"epoch_{epoch}_val", eval_mode)
        improvement_dict, epochs_since_improvement = update_best_metrics(best_metrics, val_metrics,
                                                                         epochs_since_improvement, eval_mode)

        # Report results
        print_val_results(val_metrics, f"Epoch {epoch} val results", eval_mode)

        if cfg["use_wandb"]:
            wandb_multitask_val_epoch_stats(best_metrics, val_metrics, tr_metrics, epoch, epoch_size,
                                            cfg["gradient_accumulation_steps"], start_time, eval_mode)
        metric_path = os.path.join(cfg["logdir"], f"epoch_{epoch}_metrics.pth")
        torch.save([tr_metrics, val_metrics], metric_path)

        checkpoint_dir = cfg["checkpoint_dir"]
        save_joint_idefics_checkpoint(model, optimizer, scheduler, epoch, (best_metrics, epochs_since_improvement),
                                      checkpoint_dir, improvement_dict, cfg["save_each_epoch"])

        if is_nan:
            print("Got a nan during training. Stopping.")
            break

        if epochs_since_improvement >= cfg["patience_cutoff"]:
            print(f'Model has not improved in the past {epochs_since_improvement} epochs')
            print('Stopping training')
            break

def train_epoch(cfg, epoch, model, optimizer, scheduler, tr_loaders, start_time, mode):
    split_dataloaders = cfg["use_separate_dataloaders"] or cfg["model_family_name"] in ["no_ds", "baseline"]
    if not split_dataloaders:
        return train_epoch_multitask(cfg, epoch, model, optimizer, scheduler, 
                                     tr_loaders, start_time, mode) 
    else:
        return train_epoch_multitask_no_ds(cfg, epoch, model, optimizer, scheduler,
                                           tr_loaders, start_time, mode)

## MULTITASK TRAIN FUNCTIONS ##
def train_epoch_multitask(cfg, epoch, model, optimizer, scheduler, tr_loaders, start_time, mode):
    main_loader = tr_loaders["main_loader"]
    model.train()
    epoch_size = len(main_loader)
    total_results = get_std_result_dict(cfg) 
    temp_results = get_std_result_dict(cfg)

    model_preds = {}
    criterion = nn.CrossEntropyLoss()
    index_to_token = main_loader.dataset.index_to_token
    for i, main_batch in enumerate(tqdm(main_loader)):
        # Perform forward passes for main batch
        epoch_end = (i+1) == len(main_loader)
        optimizer_step = (i+1) % (cfg["gradient_accumulation_steps"] / 2) == 0
        main_l_returns, main_s_returns = multitask_train_step(
            cfg, model, main_batch, criterion, index_to_token,
            optimizer, scheduler, epoch_end, optimizer_step
        ) 
        update_metrics(total_results, main_l_returns, main_s_returns, main_batch, "main") 
        update_metrics(temp_results, main_l_returns, main_s_returns, main_batch, "main")
        save_multitask_preds(model_preds, main_l_returns, main_s_returns, main_batch[-1])

        # Record metrics
        if cfg["use_wandb"] and i % (5 * (cfg["gradient_accumulation_steps"] // 2)) == 0:
            wandb_multitask_tr_epoch_stats(temp_results, epoch, epoch_size, cfg["gradient_accumulation_steps"],
                                           i, start_time, optimizer)

        is_nan = check_for_nans(model_preds) 
        if is_nan:
            break

    # Save outputs
    savepath = os.path.join(cfg["logdir"], f'epoch_{epoch}_train_standard_preds.pth')
    torch.save(model_preds, savepath)

    process_total_results(total_results)
    return total_results, is_nan

def train_epoch_multitask_no_ds(cfg, epoch, model, optimizer, scheduler, tr_loaders, start_time, mode):
    main_listener_loader = tr_loaders["main_listener_loader"]
    model.train()
    epoch_size = len(main_listener_loader)
    total_results = get_std_result_dict(cfg)
    temp_results = get_std_result_dict(cfg)

    model_preds = {}
    criterion = nn.CrossEntropyLoss()
    index_to_token = main_listener_loader.dataset.index_to_token
    for i, main_listener_batch in enumerate(tqdm(main_listener_loader)):
        # Perform forward passes for main batch
        main_l_returns = regular_listener_step(cfg, model, main_listener_batch, criterion, index_to_token,
                                               optimizer, scheduler) 
        main_l_returns["loss"] = main_l_returns["loss"].cpu().item() * main_l_returns["B"]

        main_speaker_batch = get_next_batch(tr_loaders, "main_speaker")
        epoch_end = (i+1) == epoch_size
        optimizer_step = (i+1) % (cfg["gradient_accumulation_steps"] / 2) == 0
        main_s_returns = regular_speaker_step(cfg, model, main_speaker_batch, criterion, epoch_end,
                                              optimizer, scheduler, optimizer_step) 
        main_s_returns["loss"] = main_s_returns["loss"].cpu().item() * main_s_returns["B"]

        update_metrics(total_results, main_l_returns, main_s_returns, None, "main") 
        update_metrics(temp_results, main_l_returns, main_s_returns, None, "main")
        save_no_ds_multitask_preds(model_preds, main_l_returns, main_s_returns,
                                   main_listener_batch[-1], main_speaker_batch[-1])

        # Record metrics
        if cfg["use_wandb"] and i % (5 * (cfg["gradient_accumulation_steps"] // 2)) == 0:
            wandb_multitask_tr_epoch_stats(temp_results, epoch, epoch_size, cfg["gradient_accumulation_steps"],
                                           i, start_time, optimizer)

    # Save outputs
    savepath = os.path.join(cfg["logdir"], f'epoch_{epoch}_train_standard_preds.pth')
    torch.save(model_preds, savepath)

    process_total_results(total_results) 
    return total_results, False

def multitask_train_step(cfg, model, batch, criterion, index_to_token,
                         optimizer, scheduler, epoch_end=False, optimizer_step=False):
    listener_returns = regular_listener_step(cfg, model, batch, criterion, index_to_token,
                                             optimizer, scheduler)
    speaker_returns = regular_speaker_step(cfg, model, batch, criterion, epoch_end,
                                           optimizer, scheduler, optimizer_step)

    B = listener_returns["B"]
    listener_returns["loss"] = listener_returns["loss"].cpu().item() * B
    speaker_returns["loss"] = speaker_returns["loss"].cpu().item() * B
    return listener_returns, speaker_returns

def regular_listener_step(cfg, model, batch, criterion, index_to_token, optimizer, scheduler):
    # Perform the forward pass
    input_tokens, attn_mask, images, image_attn_mask, target_label, \
        reward, saved_log_prob, c_mask, gt_label = batch[0]

    # Push to device
    device = model.get_listener().device
    input_tokens = input_tokens.to(device)
    attn_mask = attn_mask.to(device)
    images = images.to(device)
    image_attn_mask = image_attn_mask.to(device)
    target_label = target_label.to(device)
    saved_log_prob = saved_log_prob.to(device)
    reward = reward.to(device)
    target_label = target_label.to(device)
    gt_label = gt_label.to(device)

    # Perform the forward pass first
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        all_logits = model.forward("comprehension",
                                   [
                                       input_tokens,
                                       attn_mask,
                                       images,
                                       image_attn_mask
                                   ])
    target_logits = filter_targets(all_logits[:, -1], index_to_token) # BxC

    # Compute loss and backward pass
    all_log_probs = F.log_softmax(target_logits, dim=1)
    target_log_probs = torch.gather(all_log_probs, 1, target_label.unsqueeze(1)).squeeze(1) # B
    with torch.no_grad():
        c_term = torch.exp(target_log_probs.detach() - saved_log_prob)
        if cfg["ips_clip"] != -1:
            c_term = torch.clamp(c_term, max=cfg["ips_clip"])
        c_term[c_mask] = 1
    overall_loss = - c_term * reward * target_log_probs
    loss = torch.mean(overall_loss)

    gradient_loss = loss / cfg["gradient_accumulation_steps"]
    gradient_loss.backward()

    B = target_logits.shape[0]
    pos_count, pos_acc, neg_count, neg_acc, neg_mismatch = cl_accuracies(
        target_logits, reward, gt_label, target_label
    )
    pos_loss, neg_loss, _, _ = get_sub_losses(overall_loss, reward)

    return {
        "B" : B,
        "loss" : loss,
        "pos_acc" : pos_acc,
        "pos_count" : pos_count,
        "pos_loss" : pos_loss,
        "neg_acc" : neg_acc,
        "neg_mismatch" : neg_mismatch,
        "neg_count" : neg_count,
        "neg_loss" : neg_loss,
        "target_logits" : target_logits.detach().cpu(),
    }

def regular_speaker_step(cfg, model, batch, criterion, epoch_end,
                         optimizer, scheduler, optimizer_step=False):
    input_tokens, attn_mask, images, image_attn_mask, target_tokens, target_mask, \
        reward, saved_log_prob, c_mask = batch[1]

    # Push to device
    device = model.get_listener().device
    input_tokens = input_tokens.to(device)
    attn_mask = attn_mask.to(device)
    images = images.to(device)
    image_attn_mask = image_attn_mask.to(device)
    target_tokens = target_tokens.to(device)
    target_mask = target_mask.to(device)
    saved_log_prob = saved_log_prob.to(device)
    reward = reward.to(device)

    # Perform the forward pass first and get logits
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        all_logits = model.forward("generation",
                                   [
                                       input_tokens,
                                       attn_mask,
                                       images,
                                       image_attn_mask
                                   ])  # BxTxV

    # Compute the log term and loss
    all_log_probs = F.log_softmax(all_logits, dim=2)
    token_log_probs = torch.gather(all_log_probs, 2, target_tokens.unsqueeze(2)).squeeze(2)
    token_log_probs = token_log_probs * target_mask
    utterance_log_probs = torch.sum(token_log_probs, dim=1)
    with torch.no_grad():
        detached_log_probs = utterance_log_probs.detach()
        c_term = torch.exp(detached_log_probs - saved_log_prob)
        if cfg["ips_clip"] != -1:
            c_term = torch.clamp(c_term, max=cfg["ips_clip"])
        c_term[c_mask] = 1
    overall_loss = - c_term * reward * utterance_log_probs
    loss = torch.mean(overall_loss)

    # Backward
    gradient_norm = -1
    unnormalized_gradient_norm = -1
    gradient_loss = loss / cfg["gradient_accumulation_steps"]
    gradient_loss.backward()

    if epoch_end or optimizer_step:
        with torch.no_grad():
            unnormalized_gradient_norm = torch.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters() if p.grad is not None])).item()
        if cfg["gradient_clip_norm"] != -1:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
        with torch.no_grad():
            gradient_norm = torch.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters() if p.grad is not None])).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    B = all_logits.shape[0]
    pos_count = torch.sum(reward == 1).item()
    neg_count = B - pos_count
    pos_loss, neg_loss, _, _ = get_sub_losses(overall_loss, reward)

    return {
        "loss" : loss,
        "pos_loss" : pos_loss,
        "neg_loss" : neg_loss,
        "B" : B,
        "pos_count" : pos_count,
        "neg_count" : neg_count,
        "logits" : all_logits.detach().cpu(),
        "targets" : target_tokens.detach().cpu(),
        "masks" : target_mask.detach().cpu(),
        "gradient_norm" : gradient_norm,
        "unnormalized_gradient_norm" : unnormalized_gradient_norm,
    }

def check_for_nans(model_preds):
    found_nan = False
    for game_id, game_dict in model_preds.items():
        for round_idx, round_dict in game_dict.items():
            if torch.any(torch.isnan(round_dict['listener_outputs'])):
                found_nan = True
            if torch.any(torch.isnan(round_dict['speaker_logits'])):
                found_nan = True
            if found_nan:
                break

    return found_nan

## METRIC HELPER FUNCTIONS ##

def get_std_result_dict(cfg):
    splits = ["main"]
    results = {
        batch_type : {
            "l" : {
                "count" : 0, "pos_count" : 0, "neg_count" : 0,
                "loss" : 0, "pos_loss" : 0, "neg_loss" : 0,
                "pos_acc" : 0, "neg_acc" : 0, "neg_mismatch" : 0, 
            },
            "s" : {
                "count" : 0, "pos_count" : 0, "neg_count" : 0,
                "loss" : 0, "pos_loss" : 0, "neg_loss" : 0
            },
            "batch_count" : 0, "gradient_norm" : 0, "unnormalized_gradient_norm" : 0
        } for batch_type in splits
    }

    return results

def update_metrics(results, listener_returns, speaker_returns, batch, batch_type):
    batch_results = results[batch_type]        
    for role, result_dict in zip(["l", "s"], [listener_returns, speaker_returns]):
        batch_results[role]["loss"] += result_dict["loss"]
        batch_results[role]["pos_loss"] += result_dict["pos_loss"]
        batch_results[role]["neg_loss"] += result_dict["neg_loss"]
        batch_results[role]["count"] += result_dict["B"]
        batch_results[role]["pos_count"] += result_dict["pos_count"]
        batch_results[role]["neg_count"] += result_dict["neg_count"]
            
    for metric in ["pos_acc", "neg_acc", "neg_mismatch"]:
        batch_results["l"][metric] += listener_returns[metric]

    if speaker_returns["gradient_norm"] != -1:
        batch_results["batch_count"] += 1
        batch_results["gradient_norm"] += speaker_returns["gradient_norm"]
        batch_results["unnormalized_gradient_norm"] += speaker_returns["unnormalized_gradient_norm"]

def process_total_results(results):
    for split, split_dict in results.items():
        for initial in ["s", "l"]:
            count = split_dict[initial]["count"]
            pos_count = split_dict[initial]["pos_count"] + 1e-8
            neg_count = split_dict[initial]["neg_count"] + 1e-8

            split_dict[initial]["loss"] /= count
            split_dict[initial]["pos_loss"] /= pos_count
            split_dict[initial]["neg_loss"] /= neg_count

        pos_count = split_dict["l"]["pos_count"] + 1e-8
        neg_count = split_dict["l"]["neg_count"] + 1e-8
        split_dict["l"]["pos_acc"] /= pos_count
        split_dict["l"]["neg_acc"] /= neg_count
        split_dict["l"]["neg_mismatch"] /= neg_count

        split_dict["gradient_norm"] /= split_dict["batch_count"]
        split_dict["unnormalized_gradient_norm"] /= split_dict["batch_count"]

def save_multitask_preds(model_preds, listener_returns, speaker_returns, added_info):
    listener_outputs = listener_returns["target_logits"]
    speaker_logits = speaker_returns["logits"]
    speaker_targets = speaker_returns["targets"]
    speaker_masks = speaker_returns["masks"]

    for i in range(listener_outputs.shape[0]):    
        game_id = added_info["game_id"][i]
        round_idx = added_info["round_index"][i]
        listener_context = [added_info["listener_context"][j][i] for j in range(10)]
        speaker_context = [added_info["speaker_context"][j][i] for j in range(10)]

        # Outputs
        listener_output = listener_outputs[i]
        speaker_logit = speaker_logits[i][speaker_masks[i]]
        speaker_target = speaker_targets[i][speaker_masks[i]]

        if game_id not in model_preds:
            model_preds[game_id] = {}
        model_preds[game_id][round_idx] = {
            "listener_context" : listener_context,
            "speaker_context" : speaker_context,
            "listener_outputs" : listener_output,
            "speaker_logits" : speaker_logit,
            "speaker_targets" : speaker_target,
        }

def save_no_ds_multitask_preds(model_preds, listener_returns, speaker_returns, listener_info, speaker_info):
    listener_outputs = listener_returns["target_logits"]
    speaker_logits = speaker_returns["logits"]
    speaker_targets = speaker_returns["targets"]
    speaker_masks = speaker_returns["masks"]

    for i in range(listener_outputs.shape[0]):    
        game_id = listener_info["game_id"][i]
        round_idx = listener_info["round_index"][i]
        listener_context = [listener_info["listener_context"][j][i] for j in range(10)]

        # Outputs
        listener_output = listener_outputs[i]
        if game_id not in model_preds:
            model_preds[game_id] = {}
        if round_idx not in model_preds[game_id]:
            model_preds[game_id][round_idx] = {
                "listener_context" : listener_context,
                "listener_outputs" : listener_output,
            }
        else:
            model_preds[game_id][round_idx]["listener_context"] = listener_context
            model_preds[game_id][round_idx]["listener_outputs"] = listener_output

    for i in range(speaker_masks.shape[0]):
        game_id = speaker_info["game_id"][i]
        round_idx = speaker_info["round_index"][i]
        speaker_context = [speaker_info["speaker_context"][j][i] for j in range(10)]

        # Outputs
        speaker_logit = speaker_logits[i][speaker_masks[i]]
        speaker_target = speaker_targets[i][speaker_masks[i]]

        if game_id not in model_preds:
            model_preds[game_id] = {}
        if round_idx not in model_preds[game_id]:
            model_preds[game_id][round_idx] = {
                "speaker_context" : speaker_context,
                "speaker_logits" : speaker_logit,
                "speaker_targets" : speaker_target,
            }
        else:
            model_preds[game_id][round_idx]["speaker_context"] = speaker_context
            model_preds[game_id][round_idx]["speaker_logits"] = speaker_logit
            model_preds[game_id][round_idx]["speaker_targets"] = speaker_target

def get_best_metrics(cfg, eval_mode):
    best_metrics = {}
    if eval_mode == "multitask":
        for metric in ["acc", "sim_acc", "loss"]:
            best_metrics[f"l_{metric}"] = cfg[f"best_l_{metric}"] 
        for metric in ["loss", "rerank_loss", "rerank_acc"]:
            best_metrics[f"s_{metric}"] = cfg[f"best_s_{metric}"]
    else:
        for prefix in ["joint", "split"]:
            for suffix in ["loss", "acc", "sim_acc"]:
                best_metrics[f"l_{prefix}_{suffix}"] = cfg[f"best_l_{prefix}_{suffix}"]
            for suffix in ["loss", "rerank_acc"]:
                best_metrics[f"s_{prefix}_{suffix}"] = cfg[f"best_s_{prefix}_{suffix}"]

    return best_metrics

def update_best_metrics(best_metrics, val_metrics, epochs_since_improvement, eval_mode):
    if eval_mode == "multitask":
        improvement_dict = {
            "acc" : False
        }

        for role, metric in [("l", "acc"), ("l", "sim_acc"), ("s", "rerank_acc")]:
            if val_metrics[role][metric] > best_metrics[f"{role}_{metric}"]:
                best_metrics[f"{role}_{metric}"] = val_metrics[role][metric]
                if metric == "acc":
                    improvement_dict[metric] = True

        # Losses
        for role, metric in [("l", "loss"), ("s", "loss"), ("s", "rerank_loss")]:
            if val_metrics[role][metric] < best_metrics[f"{role}_{metric}"]:
                best_metrics[f"{role}_{metric}"] = val_metrics[role][metric]

        has_improvement = any(list(improvement_dict.values()))
        epochs_since_improvement = 0 if has_improvement else epochs_since_improvement + 1 
    else:
        improvement_dict = {
            "acc" : False
        }

        for prefix in ["joint", "split"]:
            # Accuracies
            for role, metric in [("l", "acc"), ("l", "sim_acc"), ("s", "rerank_acc")]:
                if val_metrics[role][f"{prefix}_{metric}"] > best_metrics[f"{role}_{prefix}_{metric}"]:
                    best_metrics[f"{role}_{prefix}_{metric}"] = val_metrics[role][f"{prefix}_{metric}"]
                    if metric == "acc" and prefix == "joint":
                        improvement_dict[metric] = True

            # Losses
            for role in ["l", "s"]:
                if val_metrics[role][f"{prefix}_loss"] < best_metrics[f"{role}_{prefix}_loss"]:
                    best_metrics[f"{role}_{prefix}_loss"] = val_metrics[role][f"{prefix}_loss"]

        has_improvement = any(list(improvement_dict.values()))
        epochs_since_improvement = 0 if has_improvement else epochs_since_improvement + 1 

    return improvement_dict, epochs_since_improvement

## RESULT REPORTING FUNCTIONS ##

def print_tr_results(metrics):
    for split, split_dict in metrics.items():
        for initial, role in [("s", "speaker"), ("l", "listener")]:
            loss = split_dict[initial]["loss"]
            pos_loss = split_dict[initial]["pos_loss"]
            neg_loss = split_dict[initial]["neg_loss"]
            print(f"{split} {role} results: loss: {loss}, pos loss: {pos_loss}, neg loss: {neg_loss}")

        l_pos_acc = split_dict["l"]["pos_acc"]
        l_neg_acc = split_dict["l"]["neg_acc"]
        l_neg_mismatch = split_dict["l"]["neg_mismatch"]
        print(f"{split} listener results: pos acc: {l_pos_acc}, neg acc: {l_neg_acc}, neg mismatch: {l_neg_mismatch}")

def print_val_results(metrics, init_message, eval_mode):
    print(init_message)
    if eval_mode == "multitask":
        l_loss = metrics["l"]["loss"]
        l_acc = metrics["l"]["acc"]
        l_sim_acc = metrics["l"]["sim_acc"]
        print(f'Listener: loss: {l_loss}, accuracy: {l_acc}, similarity block accuracy: {l_sim_acc}')
        
        s_loss = metrics["s"]["loss"]
        s_rerank_acc =  metrics["s"]["rerank_acc"]
        s_rerank_loss =  metrics["s"]["rerank_loss"]
        print(f"Speaker: loss {s_loss}, rerank accuracy {s_rerank_acc}, rerank loss: {s_rerank_loss}")
    else:
        for inf in ["split", "joint"]:
            # Report listener stats
            l_loss = metrics['l'][f'{inf}_loss']
            l_acc = metrics['l'][f'{inf}_acc']
            l_sim_acc = metrics['l'][f'{inf}_sim_acc']
            print(f'{inf} listener: loss: {l_loss}, accuracy: {l_acc}, similarity block accuracy: {l_sim_acc}')

            # Report speaker stats
            s_loss = metrics['s'][f'{inf}_loss']
            s_rerank_acc = metrics['s'][f'{inf}_rerank_acc']
            print(f'{inf} speaker: loss: {s_loss}, reranking accuracy: {s_rerank_acc}')

            # Report split only stats
            if inf == "split":
                print(f"Listener reranking accuracy: {metrics['l']['listener_rerank']}")
                print(f"Speaker comprehension accuracy: {metrics['s']['speaker_acc']}, similarity block accuracy: {metrics['s']['speaker_sim_acc']}")     

    print()

def wandb_multitask_tr_epoch_stats(result_dict, epoch, epoch_size, grad_steps, i, start_time, optimizer):
    time_elapsed = (time() - start_time) / 60
    curr_dict = {
        "utils/lr" : optimizer.param_groups[0]["lr"],
        "utils/time_elapsed" : time_elapsed
    }

    # Assign values and reset the temp dict
    for split, sub_dict in result_dict.items():
        # Handle losses first
        for initial, role in [("l", "listener"), ("s", "speaker")]:
            count = sub_dict[initial]["count"]
            pos_count = sub_dict[initial]["pos_count"] + 1e-8
            neg_count = sub_dict[initial]["neg_count"] + 1e-8

            curr_dict[f"{split}_{role}_train/minibatch_loss"] = sub_dict[initial]["loss"] / count
            curr_dict[f"{split}_{role}_train/minibatch_pos_loss"] = sub_dict[initial]["pos_loss"] / pos_count
            curr_dict[f"{split}_{role}_train/minibatch_neg_loss"] = sub_dict[initial]["neg_loss"] / neg_count

        # Then handle other metrics
        pos_count = sub_dict["l"]["pos_count"] + 1e-8
        neg_count = sub_dict["l"]["neg_count"] + 1e-8
        curr_dict[f"{split}_listener_train/minibatch_pos_acc"] = sub_dict["l"]["pos_acc"] / pos_count
        curr_dict[f"{split}_listener_train/minibatch_neg_acc"] = sub_dict["l"]["neg_acc"] / neg_count
        curr_dict[f"{split}_listener_train/minibatch_neg_mismatch"] = sub_dict["l"]["neg_mismatch"] / neg_count
        curr_dict[f"{split}_listener_train/gradient_norm"] = sub_dict["gradient_norm"] / (sub_dict["batch_count"] + 1e-8)
        curr_dict[f"{split}_listener_train/unnormalized_gradient_norm"] = sub_dict["unnormalized_gradient_norm"] / (sub_dict["batch_count"] + 1e-8)
    
        for role in ["l", "s"]:
            for key in sub_dict[role]:
                sub_dict[role][key] = 0
        sub_dict["batch_count"] = 0
        sub_dict["gradient_norm"] = 0
        sub_dict["unnormalized_gradient_norm"] = 0        

    grad_steps = grad_steps // 2
    epoch_steps = math.ceil(epoch_size / grad_steps) 
    curr_step = epoch_steps*epoch + i // grad_steps
    wandb.log(curr_dict, step=curr_step)

def wandb_multitask_val_epoch_stats(best_metrics, val_metrics, tr_metrics, epoch,
                                    epoch_size, grad_steps, start_time, eval_mode):
    time_elapsed = (time() - start_time) / 60
    curr_dict = {"utils/time_elapsed" : time_elapsed}

    if eval_mode == "multitask":
        # Handle the validation metrics first
        for metric in ["loss", "acc", "sim_acc"]:
            curr_dict[f"listener_val/{metric}"] = val_metrics["l"][metric]
            curr_dict[f"listener_val/best_{metric}"] = best_metrics[f"l_{metric}"]
        for metric in ["loss", "rerank_loss", "rerank_acc"]:
            curr_dict[f"speaker_val/{metric}"] = val_metrics["s"][metric]
            curr_dict[f"speaker_val/best_{metric}"] = best_metrics[f"s_{metric}"]

        # Alternative reranking metrics
        for metric in ["soft_rerank_argmax", "alt_split_rerank_acc"]:
            curr_dict[f"speaker_val/{metric}"] = val_metrics["s"][metric]
    else:
        # Validation metrics
        for inf in ["split", "joint"]:
            # Core listener metrics
            for postfix in ["loss", "acc", "sim_acc"]:
                curr_dict[f"{inf}_listener_val/{postfix}"] = val_metrics["l"][f"{inf}_{postfix}"]
                curr_dict[f"{inf}_listener_val/best_{postfix}"] = best_metrics[f"l_{inf}_{postfix}"]

            # Core speaker metrics
            for postfix in ["loss", "rerank_acc"]:
                curr_dict[f"{inf}_speaker_val/{postfix}"] = val_metrics["s"][f"{inf}_{postfix}"]
                curr_dict[f"{inf}_speaker_val/best_{postfix}"] = best_metrics[f"s_{inf}_{postfix}"]

            # Split only metrics
            if inf == "split":
                curr_dict["split_listener_val/listener_rerank"] = val_metrics["l"]["listener_rerank"]
                curr_dict["split_speaker_val/speaker_acc"] = val_metrics["s"]["speaker_acc"]
                curr_dict["split_speaker_val/speaker_sim_acc"] = val_metrics["s"]["speaker_sim_acc"]
                curr_dict["split_speaker_val/soft_rerank_argmax"] = val_metrics["s"]["soft_rerank_argmax"]
                curr_dict["split_speaker_val/alt_split_rerank_acc"] = val_metrics["s"]["alt_split_rerank_acc"]

    # Handle training metrics second
    for split, sub_dict in tr_metrics.items():
        for loss_name in ["loss", "pos_loss", "neg_loss"]:
            curr_dict[f"{split}_speaker_train/{loss_name}"] = sub_dict["s"][loss_name]
        for metric in ["loss", "pos_loss", "neg_loss", "pos_acc", "neg_acc", "neg_mismatch"]:
            curr_dict[f"{split}_listener_train/{metric}"] = sub_dict["l"][metric]

    grad_steps = grad_steps // 2
    epoch_steps = math.ceil(epoch_size / grad_steps)
    wandb.log(curr_dict, step=(epoch+1)*epoch_steps)
        
def main():
    # Get experiment arguments
    cfg = get_config()
    mode = cfg["training_type"]
    eval_mode = cfg["evaluation_type"]

    # Setup the experiment folders
    setup_cl_experiment(cfg)
    if cfg["use_wandb"]:
        setup_wandb(cfg)
    print("Init env")

    # Get the datasets and model
    tr_loaders = get_cl_idefics_loaders(cfg, "train", mode)
    val_loader = get_idefics_loader(cfg, "val", eval_mode, "standard", regular_evaluation=True)
    model = get_cl_idefics_joint_model(cfg, mode, eval_mode)
    optimizer, scheduler = get_idefics_optimizer_and_scheduler(cfg, model)
    print("Got model-optim-scheduler triplet")

    train(cfg, model, optimizer, scheduler, tr_loaders, val_loader, mode, eval_mode)

if __name__ == "__main__":
    main()
