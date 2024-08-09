import argparse
import torch
import torch.nn as nn
import wandb
import os

from continual_learning.evaluation_loops import eval_epoch_gen, eval_epoch_gen_joint
from models.model_utils import load_joint_idefics
from data_utils.dataset import get_idefics_loader
from utils.utils import construct_config, add_training_arguments, add_cl_experiment_arguments, setup_cl_experiment

TREATMENTS = ["baseline", "full", "no_ji", "no_ps", "no_ds"]

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
    
    parser.add_argument('--file_suffix', type=str, default="",
                        help="Suffix")
    parser.add_argument('--best_metric', type=str, default="acc",
                        help="Suffix")
    parser.add_argument('--alternative_model_family_name', type=str, default="",
                        help="Only to be used for the final round, when deploying the initial model")
    parser.add_argument('--alternative_deployment_round', type=int, default=-1,
                        help="Only to be used for the final round, when deploying the initial model")

    args = parser.parse_args()
    config = construct_config(args, "idefics_cl_joint_training.yaml")

    return config

def main():
    # Get experiment arguments
    cfg = get_config()
    setup_cl_experiment(cfg, initial_setup=False)

    # Get the dataloader
    logdir = cfg["logdir"]
    eval_mode = cfg["evaluation_type"]
    debug_loader = get_idefics_loader(cfg, "val", eval_mode, "standard", precompute_gen=True)

    # Load the model
    best_metric = "acc"
    model, _, _ = load_joint_idefics(cfg["logdir"], cfg["checkpoint_dir"],
                                     load_best=True, best_metric=best_metric,
                                     overwrite_lambda=True, replacement_l_lambda=0.5,
                                     replacement_s_lambda=0)
    print(f"Loaded the joint model with metric {best_metric}")
    print(cfg["logdir"])
    print(cfg["checkpoint_dir"])

    savename = "next_round_preds"
    if cfg['file_suffix'] != "":
        savename += cfg['file_suffix']

    if eval_mode == "multitask":
        eval_epoch_gen(cfg, model, debug_loader, logdir, savename)
    else:
        eval_epoch_gen_joint(cfg, model, debug_loader, logdir, savename)
        
if __name__ == "__main__":
    main()

