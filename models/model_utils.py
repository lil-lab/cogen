# File: model_utils
# -----------------
# Contain utilities for models, such as loading and saving models

import torch
import os
import transformers

from transformers import Idefics2ForConditionalGeneration
from peft import LoraConfig, get_peft_model

from utils.utils import load_arguments, mkdir, load_yaml
from models.joint_inference import IdeficsJointInferenceModel

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

## IDEFICS initialization ##

def initialize_idefics(cfg):
    # Initialize the model
    checkpoint = "HuggingFaceM4/idefics2-8b"
    model = Idefics2ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).cuda()
    if "no_lora" in cfg and cfg["no_lora"]:
        return model

    # Add LoRA adapters
    if cfg["lora_subset"] == "all":
        target_modules=r'.*(text_model|vision_model|modality_projection|perceiver_resampler).*(out_proj|fc1|fc2|down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$'
    elif cfg["lora_subset"] == "theirs": # from IDEFICS-2 tutorial notebook
        target_modules=r'.*(text_model|vision_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$'
    elif cfg["lora_subset"] == "vision_resampler":
        target_modules=r'(.*(vision_model|modality_projection|perceiver_resampler).*(out_proj|fc1|fc2|down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)|(.*(k_proj|q_proj|v_proj).*$)'
    elif cfg["lora_subset"] == "standard":
        target_modules=r'.*(k_proj|q_proj|v_proj).*$'        
    lora_config = LoraConfig(
        r=cfg["lora_r"], lora_alpha=8,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights="gaussian"
    )

    model = get_peft_model(model, lora_config)
    return model

def initialize_config_losses(cfg, mode, eval_mode=None):
    if eval_mode == "multitask":
        cfg["best_l_loss"] = float('inf')
        cfg["best_l_acc"] = -1
        cfg["best_l_sim_acc"] = -1
        cfg["best_s_loss"] = float('inf')
        cfg["best_s_rerank_acc"] = -1
        cfg["best_s_rerank_loss"] = float('inf')
        cfg["best_s_listener_acc"] = -1
    elif eval_mode == "joint":
        for inf in ["split", "joint"]:
            cfg[f"best_l_{inf}_loss"] = float('inf')
            cfg[f"best_l_{inf}_acc"] = -1
            cfg[f"best_l_{inf}_sim_acc"] = -1
            cfg[f"best_s_{inf}_loss"] = float('inf')
            cfg[f"best_s_{inf}_rerank_acc"] = -1
        cfg["best_s_listener_acc"] = -1
            
    cfg["epochs_since_improvement"] = 0
    cfg["start_epoch"] = 0
    
## Model loading ##

def load_joint_idefics(args_dir, checkpoint_dir, load_best=True, best_metric="acc",
                       overwrite_lambda=False, replacement_l_lambda=0.5, replacement_s_lambda=0.5):
    # Get the model arguments
    model_cfg = load_arguments(args_dir)
    listener_lambda = model_cfg["listener_lambda"] if "listener_lambda" in model_cfg else 0.5
    speaker_lambda = model_cfg["speaker_lambda"] if "speaker_lambda" in model_cfg else 0.5

    # Get metrics
    checkpoint = os.path.join(checkpoint_dir, best_metric) if load_best else checkpoint_dir
    metric_path = os.path.join(checkpoint, "saved_metrics.pth")
    epoch, packaged_info = torch.load(metric_path)

    if "shared_parameters" not in model_cfg or model_cfg["shared_parameters"]:
        model = initialize_idefics(model_cfg)
        model.load_adapter(checkpoint, model.active_adapter)
        model = IdeficsJointInferenceModel(listener_lambda, speaker_lambda, model=model)
    else:
        listener_model = initialize_idefics(model_cfg)
        listener_folder = os.path.join(checkpoint, "listener")
        listener_model.load_adapter(listener_folder, listener_model.active_adapter)

        speaker_model = initialize_idefics(model_cfg)
        speaker_folder = os.path.join(checkpoint, "speaker")
        speaker_model.load_adapter(speaker_folder, speaker_model.active_adapter)

        model = IdeficsJointInferenceModel(listener_lambda, speaker_lambda,
                                           listener=listener_model, speaker=speaker_model)

    if overwrite_lambda:
        model.l_lambda = replacement_l_lambda
        model.s_lambda = replacement_s_lambda

    return model, epoch, packaged_info

## Model saving ##

def save_joint_idefics_checkpoint(model, optimizer, scheduler, epoch, packaged_info, checkpoint_dir, improvement_dict,
                                  save_each_epoch=False):
    # Save the metrics
    torch.save([epoch, packaged_info], os.path.join(checkpoint_dir, 'saved_metrics.pth'))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f'latest_optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, f'latest_scheduler.pt'))
    if model.has_shared_parameters:
        model.model.save_pretrained(checkpoint_dir)
    else:
        listener_subfolder = os.path.join(checkpoint_dir, "listener")
        mkdir(listener_subfolder)
        model.listener.save_pretrained(listener_subfolder)        

        speaker_subfolder = os.path.join(checkpoint_dir, "speaker")
        mkdir(speaker_subfolder)
        model.speaker.save_pretrained(speaker_subfolder)                

    for metric_name, curr_best in improvement_dict.items():
        if curr_best:
            subfolder = os.path.join(checkpoint_dir, metric_name)
            mkdir(subfolder)

            torch.save([epoch, packaged_info], os.path.join(subfolder, 'saved_metrics.pth'))
            torch.save(optimizer.state_dict(), os.path.join(subfolder, f'best_optimizer.pt'))
            torch.save(scheduler.state_dict(), os.path.join(subfolder, f'best_scheduler.pt'))

            if model.has_shared_parameters:
                model.model.save_pretrained(subfolder)
            else:
                listener_subfolder = os.path.join(subfolder, "listener")
                mkdir(listener_subfolder)
                model.listener.save_pretrained(listener_subfolder)        

                speaker_subfolder = os.path.join(subfolder, "speaker")
                mkdir(speaker_subfolder)
                model.speaker.save_pretrained(speaker_subfolder)

    if save_each_epoch:
        subfolder = os.path.join(checkpoint_dir, f"{epoch}")
        mkdir(subfolder)

        torch.save([epoch, packaged_info], os.path.join(subfolder, 'saved_metrics.pth'))
        torch.save(optimizer.state_dict(), os.path.join(subfolder, f'best_optimizer.pt'))
        torch.save(scheduler.state_dict(), os.path.join(subfolder, f'best_scheduler.pt'))
        
        if model.has_shared_parameters:
            model.model.save_pretrained(subfolder)
        else:
            listener_subfolder = os.path.join(subfolder, "listener")
            mkdir(listener_subfolder)
            model.listener.save_pretrained(listener_subfolder)        

            speaker_subfolder = os.path.join(subfolder, "speaker")
            mkdir(speaker_subfolder)
            model.speaker.save_pretrained(speaker_subfolder)
    
## IDEFICS CONTINUAL LEARNING CODE ##
def get_cl_idefics_joint_model(cfg, mode, eval_mode):
    if cfg["shared_parameters"]:
        return get_cl_joint_shared_idefics(cfg, mode, eval_mode)
    else:
        return get_cl_joint_sep_idefics(cfg, mode, eval_mode)

def get_cl_joint_shared_idefics(cfg, mode, eval_mode):
    model = initialize_idefics(cfg)
    initialize_config_losses(cfg, mode, eval_mode)

    # Load past model
    if not cfg["from_scratch"]:
        checkpoint = os.path.join(cfg["past_checkpoint_dir"], "acc")
        model.load_adapter(checkpoint, model.active_adapter)

    if cfg["load_from_checkpoint"]:
        checkpoint = cfg["checkpoint_dir"]
        model.load_adapter(checkpoint, model.active_adapter)

        metric_path = os.path.join(cfg["checkpoint_dir"], "saved_metrics.pth")
        epoch, packaged_info = torch.load(metric_path)

        training_state, epochs_since_improvement = packaged_info
        for key, value in training_state.items():
            cfg[f"best_{key}"] = value
        cfg["epochs_since_improvement"] = epochs_since_improvement
        cfg["start_epoch"] = epoch + 1

    model = IdeficsJointInferenceModel(cfg["listener_lambda"], cfg["speaker_lambda"], model=model)
    return model
    
def get_cl_joint_sep_idefics(cfg, mode, eval_mode):
    '''
    Deprecated: From initial experiments where whether to share parameters was an option
    '''
    listener_model = initialize_idefics(cfg)
    speaker_model = initialize_idefics(cfg)
    initialize_config_losses(cfg, mode, eval_mode)

    # Load past model
    if not cfg["from_scratch"]:
        listener_checkpoint = os.path.join(cfg["past_checkpoint_dir"], "acc", "listener")
        listener_model.load_adapter(listener_checkpoint, listener_model.active_adapter)
        speaker_checkpoint = os.path.join(cfg["past_checkpoint_dir"], "acc", "speaker")
        speaker_model.load_adapter(speaker_checkpoint, speaker_model.active_adapter)

    if cfg["load_from_checkpoint"]:
        checkpoint = cfg["checkpoint_dir"]
        listener_model.load_adapter(os.path.join(checkpoint, "listener"), listener_model.active_adapter)
        speaker_model.load_adapter(os.path.join(checkpoint, "speaker"), speaker_model.active_adapter)
        
        metric_path = os.path.join(cfg["checkpoint_dir"], "saved_metrics.pth")
        epoch, packaged_info = torch.load(metric_path)
    
        training_state, epochs_since_improvement = packaged_info
        for key, value in training_state.items():
            cfg[f"best_{key}"] = value
        cfg["epochs_since_improvement"] = epochs_since_improvement
        cfg["start_epoch"] = epoch + 1

    model = IdeficsJointInferenceModel(cfg["listener_lambda"], cfg["speaker_lambda"],
                                       listener=listener_model, speaker=speaker_model)
    return model
