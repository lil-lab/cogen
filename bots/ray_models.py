# File: ray_models
# ----------------
# Ray Actor wrappers around models

from models.model_utils import load_joint_idefics
from data_utils.dataset import process_images, construct_listener_full_prompt, construct_speaker_full_prompt, \
    construct_speaker_base_prompt
from transformers import AutoProcessor
import torchvision.transforms as transforms
import pickle
import torch
import os
from PIL import Image

import asyncio
import ray

@ray.remote(num_gpus=1)
class RayModel():

    def __init__(self, cfg):
        self.treatment_name = cfg["treatment_name"]

        self.initialize_data_processing(cfg)
        self.load_model(cfg)
        self.record_inference_details(cfg)

    def initialize_data_processing(self, cfg):
        self.img_dir = os.path.join(cfg["data_prefix"], cfg["img_dir"])

        checkpoint = "HuggingFaceM4/idefics2-8b"
        self.processor = AutoProcessor.from_pretrained(checkpoint, do_image_splitting=False,
                                                       size={"longest_edge": 448, "shortest_edge": 224})

        self.comprehension_prompt = "verbose_instruction"
        self.generation_prompt = "information_after"

    def load_model(self, cfg):
        folder = os.path.join(cfg["data_prefix"], cfg["base_folder"], cfg["j_experiments_folder"], cfg["j_experiment_name"])
        args_dir, model_dir = os.path.join(folder, "logging"), os.path.join(folder, "checkpoints")
        self.model, _, _ = load_joint_idefics(args_dir, model_dir, load_best=True, best_metric="acc",
                                              overwrite_lambda=True, replacement_l_lambda=cfg["listener_lambda"],
                                              replacement_s_lambda=cfg["speaker_lambda"])

        self.model.eval()
        self.device = self.model.get_listener().device
        print(self.device, self.treatment_name)

    def record_inference_details(self, cfg):
        self.joint_inference = cfg["joint_inference"]

        # Dataset details
        with open(os.path.join(cfg["data_prefix"], cfg["index_to_token_path"]), 'rb') as f:
            self.index_to_token = pickle.load(f)
        self.img_dir = os.path.join(cfg["data_prefix"], cfg["img_dir"])
        self.base_speaker_len = cfg["base_speaker_len"]

        # Generation inference details
        self.sampling_type = cfg["sampling_type"]
        self.max_steps = cfg["max_steps"]
        self.temperature = cfg["temperature"]
        self.top_k = cfg["top_k"] # HF default
        self.top_p = cfg["top_p"] 
        self.repetition_penalty = cfg["repetition_penalty"]
        self.num_samples = cfg["num_samples"]

    async def listener_predict(self, image_paths, description):
        image_paths = [path["path"][:-4] for path in image_paths]
        if self.joint_inference:
            return await self.joint_listener_predict(image_paths, description)
        else:
            return await self.split_listener_predict(image_paths, description)

    async def speaker_predict(self, image_paths, target_path):
        image_paths = [path["path"][:-4] for path in image_paths]
        if self.joint_inference:
            return await self.joint_speaker_predict(image_paths, target_path)
        else:
            return await self.split_speaker_predict(image_paths, target_path)
    
    async def split_listener_predict(self, image_paths, description):
        # Data
        input_tokens, attn_mask, images, image_attn_mask = self.split_listener_input(image_paths, description)

        with torch.no_grad():
            # Forward
            log_probs = self.model.split_comprehension_side(input_tokens, attn_mask, images,
                                                            image_attn_mask, self.index_to_token)
            target_idx = log_probs[0].argmax().item()

        return image_paths[target_idx] + ".svg"

    async def joint_listener_predict(self, image_paths, description):
        # Data
        images, l_input_tokens, l_attn_mask, l_image_attn_mask, s_input_tokens, s_attn_mask, \
            s_image_attn_mask, s_target_mask, s_target_label = self.joint_listener_input(image_paths, description)

        with torch.no_grad():
            # Forward
            _, _, joint_log_probs = self.model.comprehension_side([
                images, l_input_tokens, l_attn_mask, l_image_attn_mask, self.index_to_token,
                s_input_tokens, s_attn_mask, s_image_attn_mask, s_target_mask, s_target_label,
            ])
            target_idx = joint_log_probs[0].argmax().item()

        return image_paths[target_idx] + ".svg"
        
    async def split_speaker_predict(self, image_paths, target_path):
        input_tokens, attn_mask, images, image_attn_mask = self.split_speaker_input(image_paths, target_path)
        
        with torch.no_grad():
            captions = self.model.split_generate(input_tokens, attn_mask, images, image_attn_mask, self.processor,
                                                 sampling_type=self.sampling_type, max_steps=self.max_steps,
                                                 temperature=self.temperature, top_k=self.top_k,
                                                 top_p=self.top_p, repetition_penalty=self.repetition_penalty,
                                                 num_samples=self.num_samples)

        caption = captions[0] # bsz=1
        return caption

    async def joint_speaker_predict(self, image_paths, target_path):
        input_tokens, attn_mask, images, image_attn_mask, label = self.joint_speaker_input(image_paths, target_path)
        
        with torch.no_grad():
            image_paths = [image_paths]
            captions, _, _, _, _ = self.model.generate(
                images, input_tokens, attn_mask, image_attn_mask, label,
                image_paths, self.processor, self.img_dir, self.index_to_token,
                max_steps=self.max_steps, sampling_type=self.sampling_type, temperature=self.temperature,
                top_k=self.top_k, top_p=self.top_p, repetition_penalty=self.repetition_penalty,
                num_samples=self.num_samples
            )

        caption = captions[0] # bsz=1
        return caption

    def split_listener_input(self, image_paths, description):
        # Get the prompt
        raw_images = process_images(self.img_dir, image_paths)
        prompt = construct_listener_full_prompt(self.processor, description, 0, self.comprehension_prompt)

        # Create the basic inputs
        outputs = self.processor(
            text=[prompt],
            images=[raw_images],
            return_tensors="pt"
        ).to(self.device)

        input_tokens = outputs['input_ids'][:, :-2]
        attn_mask = outputs['attention_mask'][:, :-2]
        attn_mask[(input_tokens == 0).bool()] = 0
        images = outputs['pixel_values']
        image_attn_mask = outputs['pixel_attention_mask']

        return input_tokens, attn_mask, images, image_attn_mask

    def joint_listener_input(self, image_paths, description):
        # Get the listener inputs
        raw_images = process_images(self.img_dir, image_paths)        
        listener_prompt = construct_listener_full_prompt(self.processor, description, 0, self.comprehension_prompt)
        outputs = self.processor(
            text=[listener_prompt],
            images=[raw_images],
            return_tensors="pt"
        ).to(self.device)

        l_input_tokens = outputs['input_ids'][:, :-2]
        l_attn_mask = outputs['attention_mask'][:, :-2]
        l_attn_mask[(l_input_tokens == 0).bool()] = 0
        images = outputs['pixel_values']
        l_image_attn_mask = outputs['pixel_attention_mask']

        # Get the speaker inputs
        prompts = []
        for i in range(10):
            prompt = construct_speaker_full_prompt(self.processor, description, i, self.generation_prompt)
            prompts.append(prompt)
        outputs = self.processor(
            text=prompts,
            images=[raw_images]*10,
            padding='longest',
            return_tensors="pt"
        ).to(self.device)

        s_input_tokens = outputs['input_ids'][:, :-1]
        s_attn_mask = outputs['attention_mask'][:, :-1]
        s_attn_mask[(s_input_tokens == 0).bool()] = 0
        s_image_attn_mask = outputs['pixel_attention_mask']
        s_target_tokens = outputs['input_ids'][:, 1:]
        s_target_mask = []
        for i in range(10):
            curr_mask = self.create_speaker_caption_mask(outputs['input_ids'][i], s_attn_mask[i])
            s_target_mask.append(curr_mask)
        s_target_mask = torch.stack(s_target_mask, dim=0)

        return images, l_input_tokens, l_attn_mask, l_image_attn_mask, s_input_tokens.unsqueeze(0), \
            s_attn_mask.unsqueeze(0), s_image_attn_mask.unsqueeze(0), s_target_mask.unsqueeze(0), \
            s_target_tokens.unsqueeze(0)

    def create_speaker_caption_mask(self, all_token_ids, text_mask):
        # Overall token comp: pad + base + caption
        padding_tokens = torch.sum(all_token_ids == 0).item()
        caption_tokens = all_token_ids.shape[0] - (padding_tokens + self.base_speaker_len)

        # Construct a mask where the last caption tokens are 1
        target_mask = torch.zeros_like(text_mask)
        target_mask[-caption_tokens:] = 1

        return target_mask.bool()
    
    def split_speaker_input(self, image_paths, target_path):
        # Get the prompt
        raw_images = process_images(self.img_dir, image_paths)
        target_idx = image_paths.index(target_path[:-4])
        base_prompt = construct_speaker_base_prompt(self.processor, target_idx, self.generation_prompt, process=True)

        # Create the basic inputs
        outputs = self.processor(
            text=[base_prompt],
            images=[raw_images],
            return_tensors="pt"
        ).to(self.device)

        input_tokens = outputs['input_ids']
        attn_mask = outputs['attention_mask']
        attn_mask[(input_tokens == 0).bool()] = 0
        images = outputs['pixel_values']
        image_attn_mask = outputs['pixel_attention_mask']

        return input_tokens, attn_mask, images, image_attn_mask

    def joint_speaker_input(self, image_paths, target_path):
        # Get the prompt
        raw_images = process_images(self.img_dir, image_paths)
        target_idx = image_paths.index(target_path[:-4])
        base_prompt = construct_speaker_base_prompt(self.processor, target_idx, self.generation_prompt, process=True)

        # Create the basic input
        outputs = self.processor(
            text=[base_prompt],
            images=[raw_images],
            return_tensors="pt"
        ).to(self.device)

        input_tokens = outputs['input_ids']
        attn_mask = outputs['attention_mask']
        attn_mask[(input_tokens == 0).bool()] = 0
        images = outputs['pixel_values']
        image_attn_mask = outputs['pixel_attention_mask']

        return input_tokens, attn_mask, images, image_attn_mask, torch.LongTensor([target_idx]).to(self.device)
        
