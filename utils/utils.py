# File: utils.py
# --------------
# Script containing various utility functions

import os
import json
import pickle
import torch
import numpy as np
import yaml
import wandb
import logging
import random

CONFIG_FOLDER = "/home/mog29/cogen/configs"

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def setup_cl_experiment(cfg, initial_setup=True):
    # Set the seed for reproducibility
    if cfg["seed"] == -1:
        cfg["seed"] = random.randint(0, 1000000)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

    # Shared variables
    model_family = cfg["model_family_name"]
    deployment_round = cfg["deployment_round"]

    # Make the relevant folders for the current experiment
    exp_name = "run" if cfg["name_suffix"] == "" else f"run_{cfg['name_suffix']}"
    cfg["expdir"] = os.path.join(
        cfg["base_folder"],
        f"r{deployment_round}_{model_family}",
        exp_name
    )
    mkdir(cfg["expdir"])
    cfg["checkpoint_dir"] = os.path.join(cfg["expdir"], 'checkpoints')
    mkdir(cfg["checkpoint_dir"])
    cfg["logdir"] = os.path.join(cfg["expdir"], 'logging')
    mkdir(cfg["logdir"])

    # Make the relevant folders for the past experiment (if finetuning past model)
    if deployment_round == 1 and model_family in ["no_ds", "baseline"]:
        past_model_family = "full" if model_family == "no_ds" else "no_ji"
    else:
        past_model_family = model_family
    past_round = deployment_round - 1 if cfg["past_round"] == -1 else cfg["past_round"]
    past_run_name = "run" if cfg["past_name_suffix"] == "" else f"run_{cfg['past_name_suffix']}"
    past_expdir = os.path.join(
        cfg["base_folder"],
        f"r{past_round}_{past_model_family}",
        past_run_name
    )
    cfg["past_checkpoint_dir"] = os.path.join(past_expdir, 'checkpoints')
    cfg["past_logdir"] = os.path.join(past_expdir, 'logging')

    if initial_setup:
        with open(os.path.join(cfg["logdir"], 'exp_cfg.yaml'), 'w') as cfg_file:
            yaml.dump(cfg, cfg_file)

def setup_wandb(cfg):
    wandb_input = {"entity" : "lil",
                   "name" : cfg["wandb_experiment_name"],
                   "project" : cfg["wandb_project_name"]}
    wandb.init(**wandb_input)

def load_pickle(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        raise(e)

def dump_pickle(filename, data):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        raise(e)

def load_json(filename):
    try:
        with open(filename) as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise(e)

def load_arguments(args_dir):
    loaded_cfg = load_yaml(os.path.join(args_dir, f'exp_cfg.yaml'))
    return loaded_cfg

## CONFIG FUNCTIONS ##
def load_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def construct_config(args, yaml_path):
    base_path = os.path.join(CONFIG_FOLDER, yaml_path)
    cfg = load_yaml(base_path)

    # Iterate over arguments and replace new arguments with defaults in the config
    args_dict = args.__dict__
    for key, value in args_dict.items():
        if value is None:
            continue
        cfg[key] = value

    return cfg

def add_cl_experiment_arguments(parser):
    # Continual learning specific
    parser.add_argument('--load_from_checkpoint', action='store_true',
                        help="If set, we will load the model from the most recent checkpoint")
    parser.add_argument('--model_family_name', type=str,
                        help="What is the prefix we use to refer to the experiments")
    parser.add_argument('--deployment_round', type=int,
                        help="Which deployment round are we training for? Starts from 0")
    parser.add_argument('--name_suffix', type=str,
                        help="Do we want to change the default experiment name?")
    parser.add_argument('--past_name_suffix', type=str,
                        help="Debugging: Do we want to load a different model from what was launched past round?")
    parser.add_argument('--past_round', type=int,
                        help="Debugging: Do we want to override the normal strategy of loading from the past round?")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Do we want to initialize models from scratch?")
    parser.add_argument('--save_each_epoch', action='store_true',
                        help="If set, save model params from each epoch")

    parser.add_argument('--lora_subset', type=str,
                        help="What subset of linear layers do we add adapters to?")
    parser.add_argument('--lora_dropout', type=float,
                        help="What dropout value to use for lora?")
    parser.add_argument('--no_lora', action='store_true',
                        help="If set, we will not perform LoRA")

    # Data loading terms
    parser.add_argument('--no_shuffling', action='store_true',
                        help="If set, we do not shuffle the context.")
    parser.add_argument('--only_seed', action='store_true',
                        help="Only use the seed data (ie: initialization setting)")
    parser.add_argument('--noise_filter', type=str,
                        help="Do we want to filter out examples?")
    parser.add_argument('--replacement_family_name', type=str,
                        help="Debugging: If we want to train a model on data that it did not collect")
    parser.add_argument('--anno_len_threshold', type=int,
                        help="If set to a positive integer, will filter out annotations longer than it due to memory constraints")
    parser.add_argument('--use_separate_dataloaders', action='store_true',
                        help="Split the shared comprehension and generation datasets into two dataloaders for data sharing")
    parser.add_argument('--listener_filter', type=str,
                        help="When doing data sharing, how should we filter the listener dataset?")
    parser.add_argument('--speaker_filter', type=str,
                        help="When doing data sharing, how should we filter the speaker dataset?")
    parser.add_argument('--ref_strat', type=str, choices=["normal", "no_ips_for_pos"],
                        help="How do we handle the IPS term?")

    # Wandb
    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, will log results to wandb.")
    parser.add_argument('--wandb_experiment_name', type=str,
                        help="The experiment name for the wandb run.")
    parser.add_argument('--wandb_project_name', type=str,
                        help="The project name for the wandb run.")

def add_experiment_arguments(parser):
    parser.add_argument('--load_from_checkpoint', action='store_true',
                        help="If set, we will load the model from the most recent checkpoint")
    parser.add_argument('--base_folder', type=str, 
                        help="The base folder in which comprehension model checkpoints will be saved")
    parser.add_argument('--experiments_folder', type=str, 
                        help='The name of the experiment family being run')
    parser.add_argument('--experiment_name', type=str, 
                        help='Name for the exact experiment being run')    
    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, will log results to wandb.")
    parser.add_argument('--wandb_experiment_name', type=str,
                        help="The experiment name for the wandb run.")
    parser.add_argument('--wandb_project_name', type=str,
                        help="The project name for the wandb run.")

def add_training_arguments(parser):
    parser.add_argument('--batch_size', type=int,
                        help="Batch size for a given batch")
    parser.add_argument('--test_batch_size', type=int,
                        help="How large is the test minibatch")
    parser.add_argument('--n_epochs', type=int,
                        help="Number of epochs to train for")
    parser.add_argument('--patience_cutoff', type=int,
                        help="How many epochs to wait before stopping training")
    parser.add_argument('--seed', type=int,
                        help="If set, will use the given seed. Otherwise randomly sample one.")
