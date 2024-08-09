# File: joint_training
# --------------------
# Training script where the speaker and listener models are jointly trained

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
import pdb

from continual_learning.evaluation_loops import full_eval_epoch

from models.model_utils import get_cl_idefics_joint_model, load_joint_idefics
from data_utils.dataset import get_idefics_loader, get_cl_idefics_loaders

from utils.utils import setup_cl_experiment, construct_config, setup_wandb
from utils.utils import add_cl_experiment_arguments, add_training_arguments

def get_config():
    parser = argparse.ArgumentParser()

    # Key arguments
    parser.add_argument('--shared_parameters', action='store_true',
                        help="If set, we will have the comprehension and generation models share parameters")
    parser.add_argument('--training_type', type=str,
                        help="Whether to perform joint training or just multi-task learning")
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
                        help="The low-rank dimension to use for QLoRA")

    # Training and experiment arguments
    add_training_arguments(parser)
    add_cl_experiment_arguments(parser)

    parser.add_argument('--eval_dataset', type=str,
                        help="What human-model interaction will we evaluate our model on?")
    parser.add_argument('--best_metric', type=str, choices=["acc", "rerank_acc"], default="acc",
                        help="According to which metric should we choose our model?")
    parser.add_argument('--initialize_from_scratch', action='store_true',
                        help="If set, we will initialize the model from scratch")

    args = parser.parse_args()
    config = construct_config(args, "idefics_cl_joint_training.yaml")

    return config

def main():
    # Get experiment arguments
    cfg = get_config()
    mode = cfg["training_type"]
    eval_mode = cfg["evaluation_type"] 

    # Setup the experiment folders
    setup_cl_experiment(cfg, initial_setup=False)
    print("Init env")

    if cfg["initialize_from_scratch"]:
        # Debugging: To test zero-shot performance for model without finetuning
        model = get_cl_idefics_joint_model(cfg, mode, eval_mode)
    else:
        best_metric = cfg["best_metric"]
        model, epoch, _ = load_joint_idefics(cfg["logdir"], cfg["checkpoint_dir"],
                                             load_best=True, best_metric=best_metric)
        print(cfg['logdir'], cfg['checkpoint_dir'])
        print(f"Loaded the model from epoch {epoch} with metric {best_metric}")

    # Get the datasets
    path = cfg['eval_dataset']
    val_loader = get_idefics_loader(cfg, "val", eval_mode, "standard", human_model_path=path)
    print(f"Loaded the {path} dataset")

    # Evaluate the model
    val_metrics = full_eval_epoch(cfg, model, val_loader, cfg["logdir"], path, eval_mode)
    print(val_metrics)

    if cfg["initialize_from_scratch"]:
        gen_prompt = cfg["generation_prompt"]
        comp_prompt = cfg["comprehension_prompt"]
        metric_path = os.path.join(cfg["logdir"], f"{path}_{gen_prompt}_{comp_prompt}_{eval_mode}_val_metrics.pth")
    else:
        metric_path = os.path.join(cfg["logdir"], f"{path}_val_metrics_{best_metric}.pth")        
    torch.save(val_metrics, metric_path)

if __name__ == "__main__":
    main()
