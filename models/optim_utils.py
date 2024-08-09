# File: optim_utils
# -----------------
# Contains utilities for initializing optimizers and schedulers for the
# various training methods

import torch
import os
import transformers
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

def get_idefics_optimizer_and_scheduler(cfg, model):
    # Get the optimizer
    optimizer = initialize_idefics_optimizer(cfg, model)
    scheduler = initialize_idefics_scheduler(cfg, optimizer)
    return optimizer, scheduler

def initialize_idefics_optimizer(cfg, opt_model):
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [
                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=cfg["learning_rate"], eps=1e-8, betas=(0.9, 0.999)
    )

    if cfg["load_from_checkpoint"]:
        load_optim_from_checkpoint(cfg, optimizer)

    return optimizer

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def load_optim_from_checkpoint(cfg, optim):
    checkpoint_path = os.path.join(cfg["checkpoint_dir"], 'latest_optimizer.pt')
    state_dict = torch.load(checkpoint_path)
    optim.load_state_dict(state_dict)

def initialize_idefics_scheduler(cfg, optimizer):
    # Use a cosine scheduler: Does not affect performance meaningfully
    num_training_steps = cfg["num_training_steps"]
    num_warmup_steps = cfg["num_warmup_steps"]
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps,
                                                             num_training_steps = num_training_steps)
    
    # Load from checkpoint if requested
    if cfg["load_from_checkpoint"]:
        checkpoint_path = os.path.join(cfg["checkpoint_dir"], 'latest_scheduler.pt')
        state_dict = torch.load(checkpoint_path)
        scheduler.load_state_dict(state_dict)

    return scheduler
