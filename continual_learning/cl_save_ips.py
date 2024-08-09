import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
from tqdm import tqdm
import pickle
import json

from transformers import AutoProcessor
from models.model_utils import load_joint_idefics
from data_utils.dataset import process_images, construct_listener_full_prompt, construct_speaker_full_prompt
from utils.utils import setup_cl_experiment, construct_config, add_training_arguments, add_cl_experiment_arguments

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

    parser.add_argument('--best_metric', type=str, choices=["acc", "rerank_acc"], default="acc",
                        help="According to which metric should we choose our model?")
    parser.add_argument('--treatment_name', type=str,
                        help="The name of the treatment to compute IPS terms for")

    args = parser.parse_args()
    config = construct_config(args, "idefics_cl_joint_training.yaml")

    return config

def prepare_processor_and_indices(cfg):
    # Get the processor and image transform
    checkpoint = "HuggingFaceM4/idefics2-8b"
    processor = AutoProcessor.from_pretrained(checkpoint, do_image_splitting=False,
                                              size={"longest_edge": 448, "shortest_edge": 224})

    # Get index to token
    with open(os.path.join("/home/mog29", "data_and_checkpoints/index_to_token.pkl"), 'rb') as f:
        index_to_token = pickle.load(f)
        
    return processor, index_to_token

def compute_ips_terms(cfg, model, processor, index_to_token, cl_examples, suffix):
    model.eval()
    with torch.no_grad():
        for game_id, game_dict in tqdm(cl_examples.items()):
            for round_index, round_dict in game_dict.items():
                ips_terms = get_model_prediction(cfg, model, processor, index_to_token, round_dict, game_id) 
                for role in ["listener", "speaker"]:
                    round_dict[f'{role}_logp'] = ips_terms[f"{role}_logp"]

    # Save the updated examples
    treatment = cfg["treatment_name"]
    round_idx = cfg["deployment_round"] + 1
    json_path = os.path.join(cfg["split_dir"], f"cl_r{round_idx}_{treatment}_{suffix}.json")
    with open(json_path, 'w') as f:
        json.dump(cl_examples, f)

def get_model_prediction(cfg, model, processor, index_to_token, round_dict, game_id):
    device = model.get_listener().device
    ips_terms = {}
    key_name = "selection" if "listener" in game_id else "gt_target"

    image_paths = round_dict["listener_context"]
    description = round_dict["chat"].lower().strip()
    if "no_ji" in game_id or "baseline" in game_id:
        input_tokens, attn_mask, images, image_attn_mask = split_listener_input(cfg, processor, image_paths, description, device) 
        with torch.no_grad():
            listener_log_probs = model.split_comprehension_side(input_tokens, attn_mask, images, image_attn_mask, index_to_token)
    else:
        images, l_input_tokens, l_attn_mask, l_image_attn_mask, s_input_tokens, s_attn_mask, \
            s_image_attn_mask, s_target_mask, s_target_token = joint_listener_input(cfg, processor, image_paths, description, device) 
        _, _, listener_log_probs = model.comprehension_side([
            images, l_input_tokens, l_attn_mask, l_image_attn_mask, index_to_token,
            s_input_tokens, s_attn_mask, s_image_attn_mask, s_target_mask, s_target_token
        ])

    # Model idled during deployment: Recompute
    if key_name == "selection" and round_dict["selection"] == "redo":
        model_pred = torch.argmax(listener_log_probs[0]).item()
        round_dict["selection"] = image_paths[model_pred]
        round_dict["reward"] = 1 if round_dict["selection"] == round_dict["gt_target"] else -1
    target_idx = image_paths.index(round_dict[key_name])
    ips_terms["listener_logp"] = listener_log_probs[0, target_idx].item()

    image_paths = round_dict['speaker_context']
    target_idx = image_paths.index(round_dict[key_name])
    input_tokens, attn_mask, images, image_attn_mask, target_tokens, \
        target_mask = split_speaker_input(cfg, processor, image_paths, description, target_idx, device) 
    with torch.no_grad():
        all_logits = model.forward("generation",
                      [
                          input_tokens,
                          attn_mask,
                          images,
                          image_attn_mask
                      ]
        )

        all_log_probs = F.log_softmax(all_logits, dim=2)
        token_log_probs = torch.gather(all_log_probs, 2, target_tokens.unsqueeze(2)).squeeze(2)
        token_log_probs = token_log_probs * target_mask
        speaker_log_probs = torch.sum(token_log_probs, dim=1)
    ips_terms["speaker_logp"] = speaker_log_probs[0].item()
    
    return ips_terms

def split_listener_input(cfg, processor, image_paths, description, device):
    # Get the prompt
    raw_images = process_images(cfg["img_dir"], image_paths)
    prompt = construct_listener_full_prompt(processor, description, 0, "verbose_instruction")

    # Create the basic inputs
    outputs = processor(
        text=[prompt],
        images=[raw_images],
        return_tensors="pt"
    ).to(device)

    input_tokens = outputs['input_ids'][:, :-2]
    attn_mask = outputs['attention_mask'][:, :-2]
    attn_mask[(input_tokens == 0).bool()] = 0
    images = outputs['pixel_values']
    image_attn_mask = outputs['pixel_attention_mask']

    return input_tokens, attn_mask, images, image_attn_mask

def joint_listener_input(cfg, processor, image_paths, description, device):
    # Get the listener inputs
    raw_images = process_images(cfg["img_dir"], image_paths)        
    listener_prompt = construct_listener_full_prompt(processor, description, 0, "verbose_instruction")

    outputs = processor(
        text=[listener_prompt],
        images=[raw_images],
        return_tensors="pt"
    ).to(device)

    l_input_tokens = outputs['input_ids'][:, :-2]
    l_attn_mask = outputs['attention_mask'][:, :-2]
    l_attn_mask[(l_input_tokens == 0).bool()] = 0
    images = outputs['pixel_values']
    l_image_attn_mask = outputs['pixel_attention_mask']

    # Get the speaker inputs
    prompts = []
    for i in range(10):
        prompt = construct_speaker_full_prompt(processor, description, i, "information_after")
        prompts.append(prompt)
    outputs = processor(
        text=prompts,
        images=[raw_images]*10,
        padding='longest',
        return_tensors="pt"
    ).to(device)

    s_input_tokens = outputs['input_ids'][:, :-1]
    s_attn_mask = outputs['attention_mask'][:, :-1]
    s_attn_mask[(s_input_tokens == 0).bool()] = 0
    s_image_attn_mask = outputs['pixel_attention_mask']
    s_target_tokens = outputs['input_ids'][:, 1:]
    s_target_mask = []
    for i in range(10):
        curr_mask = create_speaker_caption_mask(outputs['input_ids'][i], s_attn_mask[i])
        s_target_mask.append(curr_mask)
    s_target_mask = torch.stack(s_target_mask, dim=0)

    return images, l_input_tokens, l_attn_mask, l_image_attn_mask, s_input_tokens.unsqueeze(0), \
        s_attn_mask.unsqueeze(0), s_image_attn_mask.unsqueeze(0), s_target_mask.unsqueeze(0), \
        s_target_tokens.unsqueeze(0)

def split_speaker_input(cfg, processor, image_paths, description, target_idx, device):
    raw_images = process_images(cfg["img_dir"], image_paths)
    prompt = construct_speaker_full_prompt(processor, description, target_idx, "information_after")

    outputs = processor(
        text=[prompt],
        images=[raw_images],
        return_tensors="pt"
    ).to(device)
    
    input_tokens = outputs['input_ids'][:, :-1]
    attn_mask = outputs['attention_mask'][:, :-1]
    attn_mask[(input_tokens == 0).bool()] = 0
    images = outputs['pixel_values']
    image_attn_mask = outputs['pixel_attention_mask']

    target_tokens = outputs['input_ids'][:, 1:]
    target_mask = create_speaker_caption_mask(outputs['input_ids'][0], attn_mask[0])

    return input_tokens, attn_mask, images, image_attn_mask, target_tokens, target_mask

def create_speaker_caption_mask(all_token_ids, text_mask):
    # Overall token comp: pad + base + caption
    padding_tokens = torch.sum(all_token_ids == 0).item()
    caption_tokens = all_token_ids.shape[0] - (padding_tokens + 787)

    # Construct a mask where the last caption tokens are 1
    target_mask = torch.zeros_like(text_mask)
    target_mask[-caption_tokens:] = 1
    return target_mask.bool().unsqueeze(0)

def load_unprocessed_json(cfg, suffix):
    treatment = cfg["treatment_name"]
    round_idx = cfg["deployment_round"] + 1

    json_path = os.path.join(cfg["split_dir"], f"cl_r{round_idx}_{treatment}_{suffix}_unprocessed.json")

    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    # Get experiment arguments
    cfg = get_config()
    setup_cl_experiment(cfg, initial_setup=False) 

    # Load the desired model
    model, _, _ = load_joint_idefics(cfg["logdir"], cfg["checkpoint_dir"], load_best=True, best_metric="acc")
    print(cfg['logdir'], cfg['checkpoint_dir'])    
    print("Loaded the joint model")

    processor, index_to_token = prepare_processor_and_indices(cfg)
    suffixes = ["listener", "speaker"] if cfg["treatment_name"] in ["baseline", "no_ds"] else ["all"] 
    for suffix in suffixes:
        example_json = load_unprocessed_json(cfg, suffix)
        compute_ips_terms(cfg, model, processor, index_to_token, example_json, suffix) 
        
if __name__ == "__main__":
    main()

