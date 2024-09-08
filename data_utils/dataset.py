# File: dataset.py
# ----------------
# Contains key dataset and data processing utilities for training/evaluation

import copy
import os
import torch
import pickle
import json
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import math

from transformers import AutoProcessor
import torchvision.transforms as transforms

import inflect
p = inflect.engine()
import pdb


TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

### DATASETS ###

class IDEFICSDataset(Dataset):
    '''
    Dataset containing a list of image-text pairs, which we will process according to
    IDEFICS' requirements
    '''

    def __init__(self, cfg, split, task, eval_scheme=None, suffix=None,
                 regular_evaluation=False, human_model_path="", debug_gen=False,
                 precompute_gen=False, precompute_analysis=False, noise_filter=""):
        # General data loader parameters
        self.img_dir = cfg["img_dir"]
        self.task = task
        self.eval_scheme = eval_scheme        
        self.anno_len_threshold = cfg["anno_len_threshold"]
        self.no_shuffling = cfg["no_shuffling"]
        self.cl_training = suffix is not None

        # IDEFICS Miscellany
        checkpoint = "HuggingFaceM4/idefics2-8b"
        self.processor = AutoProcessor.from_pretrained(checkpoint, do_image_splitting=False,
                                                       size={"longest_edge": 448, "shortest_edge": 224})

        self.generation_prompt_type = cfg["generation_prompt"]
        self.comprehension_prompt_type = cfg["comprehension_prompt"]

        # Preprocessing
        if self.cl_training:
            print("Loading without rehearsal")
            self.load_all_rounds(cfg, suffix, noise_filter) 
        else:
            self.load_eval_dataset(cfg, split, eval_scheme, regular_evaluation, human_model_path,
                                   debug_gen, precompute_gen, precompute_analysis)

        self.initialize_speaker_lens()
        self.max_speaker_len = self.get_max_speaker_len()
        self.max_listener_len = self.get_max_listener_len()
        self.filter_distractors(cfg)
        self.max_distractor_count = self.get_max_distractor_count()
        self.index_to_token = self.get_listener_target_to_token()

    def __len__(self):
        return len(self.context_list)

    # Preprocessing

    def load_all_rounds(self, cfg, suffix, noise_filter):
        self.context_list = []
        if cfg["only_seed"]:
            self.append_seed_data(cfg)
        else:
            for i in range(cfg["deployment_round"] + 1):        
                if i == 0:
                    self.append_seed_data(cfg)
                else:
                    self.append_particular_round(cfg, suffix, i, noise_filter)
        
    def skip_noise(self, cfg, game_id, curr_round, noise_filter):
        if noise_filter == "":
            return False
        elif noise_filter == "only_pos":
            pos_reward = curr_round['reward'] == 1
            return not pos_reward
        elif noise_filter == "no_neg_gen":
            gen_round = "speaker" in game_id
            neg_reward = curr_round["reward"] == -1
            return gen_round and neg_reward
        elif noise_filter == "no_gen":
            gen_round = "speaker" in game_id
            return gen_round
        elif noise_filter == "no_neg_comp":
            comp_round = "listener" in game_id
            neg_reward = curr_round["reward"] == -1
            return comp_round and neg_reward
        elif noise_filter == "no_comp":
            comp_round = "listener" in game_id
            return comp_round
        else:
            assert(False)

    def skip_datapoint(self, curr_round):
        if self.eval_scheme is None and self.anno_len_threshold != -1:
            anno = curr_round["chat"].lower().strip()
            ids = self.processor([anno], return_tensors="pt")['input_ids']
            if ids.shape[-1] > self.anno_len_threshold:
                return True

        return False

    def append_particular_round(self, cfg, suffix, round_idx, noise_filter):
        model_family = cfg["model_family_name"] if cfg["replacement_family_name"] == "" else cfg["replacement_family_name"]
        filename = f"cl_r{round_idx}_{model_family}_{suffix}.json"            
        json_path = os.path.join(cfg["split_dir"], filename)
        print(f"Loading data from: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for game_id, round_dict in data.items():
            for round_index, curr_round in round_dict.items():
                if self.skip_datapoint(curr_round):
                    continue
                if self.skip_noise(cfg, game_id, curr_round, noise_filter):
                    continue
        
                context_dict = {
                    "speaker_context" : curr_round["speaker_context"],
                    "listener_context" : curr_round["listener_context"],
                    "gt_target" : curr_round["gt_target"],
                    "selection" : curr_round["selection"],
                    "target_anno" : curr_round["chat"],
                    "distractor_annos" : curr_round["distractor_annos"],
                    "similarity_block" : curr_round["similarity_block"],
                    "game_id" : game_id,
                    "round_index" : round_index,
                    "round" : curr_round["round"],
                    "reward" : curr_round["reward"],
                    "listener_logp" : curr_round["listener_logp"],
                    "speaker_logp" : curr_round["speaker_logp"],
                }

                self.context_list.append(context_dict)

    def append_seed_data(self, cfg):
        filename = "seed_train.json"
        json_path = os.path.join(cfg["split_dir"], filename)
        print('Loading data from: ', json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)

        for game_id, round_dict in data.items():
            for round_index, curr_round in round_dict.items():
                if self.skip_datapoint(curr_round):
                    continue
        
                assert(curr_round['gt_target'] == curr_round['selection'])
                context_dict = {
                    "speaker_context" : curr_round["speaker_context"],
                    "listener_context" : curr_round["listener_context"],
                    "gt_target" : curr_round["gt_target"],
                    "selection" : curr_round["selection"],
                    "target_anno" : curr_round["chat"],
                    "distractor_annos" : curr_round["distractor_annos"],
                    "similarity_block" : curr_round["similarity_block"],
                    "game_id" : game_id,
                    "round_index" : round_index,
                    "round" : 0,
                    "reward" : 1,
                    "listener_logp" : 0,
                    "speaker_logp" : 0,
                }

                self.context_list.append(context_dict)

    def load_eval_dataset(self, cfg, split, eval_scheme, regular_evaluation, human_model_path,
                          debug_gen, precompute_gen, precompute_analysis):
        self.context_list = []
        if regular_evaluation: 
            self.load_dataset_validation(cfg)
        elif human_model_path != "": 
            self.load_dataset_hm(cfg, human_model_path)
        elif debug_gen:
            self.load_dataset_debug(cfg)
        elif precompute_gen:
            self.load_dataset_precompute(cfg) 
        elif precompute_analysis:
            self.load_dataset_analysis(cfg)
        else:
            assert(False)

    def load_dataset_validation(self, cfg):
        path_list = []
        filename = 'seed_val.json'
        json_path = os.path.join(cfg["split_dir"], filename)
        path_list.append(json_path)

        for path in path_list:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for game_id, round_dict in data.items():
                for round_index, curr_round in round_dict.items():
                    context_dict = {
                        "speaker_context" : curr_round["speaker_context"],
                        "listener_context" : curr_round["listener_context"],
                        "gt_target" : curr_round["gt_target"],
                        "selection" : curr_round["selection"],
                        "target_anno" : curr_round["chat"],
                        "distractor_annos" : curr_round["distractor_annos"],
                        "similarity_block" : curr_round["similarity_block"],
                        "game_id" : game_id,
                        "round_index" : round_index
                    }
                    self.context_list.append(context_dict)

    def load_dataset_hm(self, cfg, human_model_path):
        # Load data from a particular human-model run
        json_path = os.path.join(cfg['split_dir'], f"{human_model_path}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Add rounds belonging to listener rounds
        for game_id, round_dict in data.items():
            if "listener" not in game_id:
                continue
            for round_index, curr_round in round_dict.items():
                context_dict = {
                    "speaker_context" : curr_round["speaker_context"],
                    "listener_context" : curr_round["listener_context"],
                    "gt_target" : curr_round["gt_target"],
                    "selection" : curr_round["selection"],
                    "target_anno" : curr_round["chat"],
                    "distractor_annos" : curr_round["distractor_annos"],
                    "similarity_block" : curr_round["similarity_block"],
                    "game_id" : game_id,
                    "round_index" : round_index
                }
                self.context_list.append(context_dict)

    def load_dataset_debug(self, cfg):
        filename = 'generation_debugging_data.json'
        json_path = os.path.join(cfg['split_dir'], filename)
        with open(json_path, 'r') as f:
            data = json.load(f)

        for game_id, round_dict in data.items():
            for round_index, curr_round in round_dict.items():
                context_dict = {
                    "speaker_context" : curr_round["speaker_context"],
                    "listener_context" : curr_round["listener_context"],
                    "gt_target" : curr_round["gt_target"],
                    "target_anno" : "Placeholder",
                    "distractor_annos" : ["Placeholder!"],
                    "game_id" : game_id,
                    "round_index" : round_index
                }

                self.context_list.append(context_dict)

    def load_dataset_precompute(self, cfg):
        if cfg["alternative_model_family_name"] != "":
            # Used only for the final round of deployment for the initial Full model
            data_round = cfg["alternative_deployment_round"]
            model_family = cfg["alternative_model_family_name"]
        else:
            data_round = cfg["deployment_round"] + 1
            model_family = cfg["model_family_name"]
        filename = f"cl_r{data_round}_{model_family}_speaker_precompute.json"
        json_path = os.path.join(cfg['split_dir'], filename)
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(json_path)

        for game_id, round_dict in data.items():
            for round_index, curr_round in round_dict.items():
                context_dict = {
                    "speaker_context" : curr_round["speaker_context"],
                    "listener_context" : curr_round["listener_context"],
                    "gt_target" : curr_round["gt_target"],
                    "target_anno" : "Placeholder",
                    "distractor_annos" : ["Placeholder!"],
                    "game_id" : game_id,
                    "round_index" : round_index
                }

                self.context_list.append(context_dict)

    def load_dataset_analysis(self, cfg):
        filename = "human_human.json"
        json_path = os.path.join(cfg["split_dir"], "analysis", filename)
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(json_path)

        for game_id, round_dict in data.items():
            for round_index, curr_round in round_dict.items():
                context_dict = {
                    "speaker_context" : curr_round["speaker_context"],
                    "listener_context" : curr_round["listener_context"],
                    "gt_target" : curr_round["gt_target"],
                    "target_anno" : "Placeholder",
                    "distractor_annos" : ["Placeholder!"],
                    "game_id" : game_id,
                    "round_index" : round_index
                }

                self.context_list.append(context_dict)


    # Miscellaneous helpers

    def initialize_speaker_lens(self):
        # Get an example context
        context_dict = self.context_list[0]
        context_images = context_dict['speaker_context']
        raw_images = process_images(self.img_dir, context_images)

        # Get prompt lengths for each target index: Check if you still get distinct lengths
        self.prompt_len_dict = {}
        max_prompt_len = -1
        self.max_prompt_idx = -1
        for target_idx in range(10):
            base_prompt = construct_speaker_base_prompt(self.processor, target_idx, self.generation_prompt_type, process=True)
            outputs = self.processor(text=[base_prompt], images=[raw_images], return_tensors="pt")
            prompt_len = outputs['input_ids'].shape[1]

            self.prompt_len_dict[target_idx] = prompt_len
            if prompt_len > max_prompt_len:
                max_prompt_len = prompt_len
                self.max_prompt_idx = target_idx

    def get_max_speaker_len(self):
        # Determine the maximum input sequence length
        max_len = -1
        for context_dict in self.context_list:
            context_images = context_dict['speaker_context']
            raw_images = process_images(self.img_dir, context_images)
            target_anno = context_dict["target_anno"]
            prompt = construct_speaker_full_prompt(self.processor, target_anno, self.max_prompt_idx, self.generation_prompt_type)
            outputs = self.processor(text=[prompt], images=[raw_images], return_tensors="pt")
            output_len = outputs['input_ids'].shape[1]
            max_len = max(max_len, output_len)

        return max_len

    def get_max_listener_len(self):
        max_len = -1
        for context_dict in self.context_list:
            context_images = context_dict['listener_context']
            raw_images = process_images(self.img_dir, context_images)
            target_anno = context_dict["target_anno"]
            target_idx = context_images.index(context_dict["gt_target"])
            prompt = construct_listener_full_prompt(self.processor, target_anno, target_idx, self.comprehension_prompt_type)
            outputs = self.processor(text=[prompt], images=[raw_images], return_tensors="pt")
            output_len = outputs['input_ids'].shape[1]
            max_len = max(max_len, output_len)

        return max_len

    def filter_distractors(self, cfg):
        for ctx in self.context_list:
            anno = ctx["target_anno"]
            filtered_distractors = [dist for dist in ctx["distractor_annos"] if dist != anno]
            ctx["distractor_annos"] = filtered_distractors

    def get_max_distractor_count(self):
        counts = [len(ctx["distractor_annos"]) for ctx in self.context_list]
        return max(counts)

    def get_listener_target_to_token(self):
        # Get an example context
        target_to_token = []
        context_dict = self.context_list[0]
        context_images = context_dict['listener_context']
        raw_images = process_images(self.img_dir, context_images)
        target_anno = context_dict["target_anno"]

        for target_idx in range(10):
            prompt = construct_listener_full_prompt(self.processor, target_anno, target_idx, self.comprehension_prompt_type)
            outputs = self.processor(text=[prompt], images=[raw_images], return_tensors="pt")
            target_to_token.append(outputs['input_ids'][0, -2].item())

        return target_to_token

    # __getitem__s
    def __getitem__(self, idx):
        if self.task == "joint":
            if self.eval_scheme is None or self.eval_scheme == "standard":
                return self.getitem_joint_standard(idx)
            elif self.eval_scheme == "generation":
                return self.getitem_joint_gen(idx)
        elif self.task == "multitask":
            if self.eval_scheme is None:
                return self.getitem_multitask(idx)
            elif self.eval_scheme == "standard":
                return self.getitem_multitask_rerank(idx)
            else:
                return self.getitem_speaker_gen(idx) 

    # Listener __getitem__
    def getitem_listener(self, idx):
        context_dict = self.context_list[idx]

        # Generate the raw image sequence
        context_images = context_dict['listener_context']
        if self.eval_scheme is None and not self.no_shuffling: 
            random.shuffle(context_images)
        raw_images = process_images(self.img_dir, context_images)

        if self.cl_training:
            # Determine what the key name is
            if context_dict['reward'] == 1 or 'speaker' in context_dict['game_id']:
                key_name = 'gt_target'
            else:
                key_name = 'selection'
        else:
            key_name = 'gt_target'
        target_idx = context_images.index(context_dict[key_name])

        # Generate the similarity block indices
        sim_idx = [context_images.index(sim_img) for sim_img in context_dict["similarity_block"]]
        if len(sim_idx) < 4:
            sim_idx = sim_idx + [sim_idx[-1]]

        # Create the prompt
        target_anno = context_dict["target_anno"]
        prompt = construct_listener_full_prompt(self.processor, target_anno, target_idx, self.comprehension_prompt_type)

        # Create the basic inputs
        outputs = self.processor(
            text=[prompt],
            images=[raw_images],
            padding='max_length',
            max_length=self.max_listener_len,
            return_tensors="pt"
        )
        input_tokens = outputs['input_ids'][0, :-2]
        attn_mask = outputs['attention_mask'][0, :-2]
        attn_mask[(input_tokens == 0).bool()] = 0
        images = outputs['pixel_values'][0]
        image_attn_mask = outputs['pixel_attention_mask'][0]
        added_info = {
            "game_id" : context_dict["game_id"],
            "round_index" : context_dict["round_index"],
            "context" : context_images
        }

        if self.cl_training:
            reward = context_dict["reward"]
            saved_log_prob = context_dict["listener_logp"]
            c_mask = reward == 1
            gt_idx = context_images.index(context_dict["gt_target"])
            return input_tokens, attn_mask, images, image_attn_mask, target_idx, \
                reward, saved_log_prob, c_mask, gt_idx, added_info
        else:
            return input_tokens, attn_mask, images, image_attn_mask, target_idx, sim_idx, added_info

    def create_speaker_caption_mask(self, all_token_ids, text_mask, target_idx):
        # Overall token comp: max_len = pad + base + caption
        padding_tokens = torch.sum(all_token_ids == 0).item()
        base_tokens = self.prompt_len_dict[target_idx]
        caption_tokens = self.max_speaker_len - (padding_tokens + base_tokens)

        # Construct a mask where the last caption tokens are 1
        target_mask = torch.zeros_like(text_mask)
        target_mask[-caption_tokens:] = 1
        return target_mask.bool()

    def getitem_speaker(self, idx):
        context_dict = self.context_list[idx]

        # Generate the raw image sequence
        context_images = context_dict['speaker_context']
        if self.eval_scheme is None and not self.no_shuffling: 
            random.shuffle(context_images)
        raw_images = process_images(self.img_dir, context_images)
        
        if self.cl_training:
            # Determine what the key name is
            if context_dict['reward'] == 1 or 'speaker' in context_dict['game_id']:
                key_name = 'gt_target'
            else:
                key_name = 'selection'
        else:
            key_name = 'gt_target'
        target_idx = context_images.index(context_dict[key_name])

        # Create the prompt
        target_anno = context_dict["target_anno"]
        prompt = construct_speaker_full_prompt(self.processor, target_anno, target_idx, self.generation_prompt_type) 

        # Create the basic inputs
        outputs = self.processor(
            text=[prompt],
            images=[raw_images],
            padding='max_length',
            max_length=self.max_speaker_len,
            return_tensors="pt"
        )
        all_token_ids = outputs['input_ids'][0] # T
        all_attn_mask = outputs['attention_mask'][0] # T
        all_attn_mask[(all_token_ids == 0).bool()] = 0
        images = outputs['pixel_values'][0] # 10x3x224x224
        image_attn_mask = outputs['pixel_attention_mask'][0] # Tx10

        input_tokens = all_token_ids[:-1]
        attn_mask = all_attn_mask[:-1]
        target_tokens = all_token_ids[1:]
        target_mask = self.create_speaker_caption_mask(all_token_ids, attn_mask, target_idx)

        added_info = {
            "game_id" : context_dict["game_id"],
            "round_index" : context_dict["round_index"],
            "context" : context_images
        }

        if self.cl_training:
            reward = context_dict["reward"]
            saved_log_prob = context_dict["speaker_logp"]
            c_mask = reward == 1
            return input_tokens, attn_mask, images, image_attn_mask, target_tokens, target_mask, \
                reward, saved_log_prob, c_mask, added_info            
        else:
            return input_tokens, attn_mask, images, image_attn_mask, target_tokens, target_mask, added_info

    def getitem_speaker_gen(self, idx):
        context_dict = self.context_list[idx]

        # Initialize the speaker inputs
        context_images = context_dict['speaker_context']        
        raw_images = process_images(self.img_dir, context_images)
        target_idx = context_images.index(context_dict['gt_target'])
        base_prompt = construct_speaker_base_prompt(self.processor, target_idx, self.generation_prompt_type, process=True)

        # Process the speaker inputs
        outputs = self.processor(
            text=[base_prompt],
            images=[raw_images],
            padding='max_length',
            max_length=self.prompt_len_dict[self.max_prompt_idx],
            return_tensors="pt"
        )
        input_tokens = outputs['input_ids'][0] # T
        attn_mask = outputs['attention_mask'][0] # T
        attn_mask[(input_tokens == 0).bool()] = 0
        images = outputs['pixel_values'][0] # 10x3x224x224
        image_attn_mask = outputs['pixel_attention_mask'][0] # Tx10
        listener_context = context_dict['listener_context']

        # Get the single reference we have for the target
        added_info = {
            "game_id" : context_dict["game_id"],
            "round_index" : context_dict["round_index"],
            "context" : context_images,
            "listener_context" : listener_context
        }

        return input_tokens, attn_mask, images, image_attn_mask, added_info

    def getitem_speaker_rerank(self, idx):
        context_dict = self.context_list[idx]

        # Generate the image inputs
        context_images = context_dict['speaker_context']
        raw_images = process_images(self.img_dir, context_images)
        target_idx = context_images.index(context_dict['gt_target'])

        # Get the annotations
        distractors = context_dict["distractor_annos"]
        pad_size = 0
        if len(distractors) < self.max_distractor_count:
            pad_size = self.max_distractor_count - len(distractors)
            distractors = distractors + [distractors[-1] for _ in range(pad_size)]
        annotations = [context_dict['target_anno']] + distractors

        # Process all examples
        prompts = []
        for anno in annotations:
            prompt = construct_speaker_full_prompt(self.processor, anno, target_idx, self.generation_prompt_type)
            prompts.append(prompt)
        outputs = self.processor(
            text=prompts,
            images=[raw_images]*len(annotations),
            padding='max_length',
            max_length=self.max_speaker_len,
            truncation=True,
            return_tensors="pt"
        )
        images = outputs['pixel_values'][0]
        input_tokens = outputs['input_ids'][:, :-1]
        attn_mask = outputs['attention_mask'][:, :-1]
        attn_mask[(input_tokens == 0).bool()] = 0
        image_attn_mask = outputs['pixel_attention_mask']
        target_tokens = outputs['input_ids'][:, 1:]
        target_mask = []
        for i in range(len(annotations)):
            curr_mask = self.create_speaker_caption_mask(outputs['input_ids'][i], attn_mask[i], target_idx)
            target_mask.append(curr_mask)
        target_mask = torch.stack(target_mask, dim=0)

        added_info = {
            "game_id" : context_dict["game_id"],
            "round_index" : context_dict["round_index"],
            "context" : context_images
        }

        return input_tokens, attn_mask, images, image_attn_mask, target_tokens, target_mask, added_info

    def getitem_joint_gen(self, idx):
        context_dict = self.context_list[idx]

        # Initialize the speaker inputs
        context_images = context_dict['speaker_context']        
        raw_images = process_images(self.img_dir, context_images)
        target_idx = context_images.index(context_dict['gt_target'])
        base_prompt = construct_speaker_base_prompt(self.processor, target_idx, self.generation_prompt_type, process=True)

        # Process the speaker inputs
        outputs = self.processor(
            text=[base_prompt],
            images=[raw_images],
            padding='max_length',
            max_length=self.prompt_len_dict[self.max_prompt_idx],
            return_tensors="pt"
        )
        input_tokens = outputs['input_ids'][0] # T
        attn_mask = outputs['attention_mask'][0] # T
        attn_mask[(input_tokens == 0).bool()] = 0
        images = outputs['pixel_values'][0] # 10x3x224x224
        image_attn_mask = outputs['pixel_attention_mask'][0] # Tx10
        listener_context = context_dict['listener_context']

        added_info = {
            "game_id" : context_dict["game_id"],
            "round_index" : context_dict["round_index"],
            "speaker_context" : context_images,
            "listener_context" : listener_context
        }

        return images, input_tokens, attn_mask, image_attn_mask, target_idx, added_info

    def getitem_joint_standard(self, idx):
        context_dict = self.context_list[idx]
        listener_batch = self.getitem_joint_standard_listener(context_dict)
        listener_batch, listener_context = listener_batch[:-1], listener_batch[-1]
        speaker_batch = self.getitem_joint_standard_speaker(context_dict)
        speaker_batch, speaker_context = speaker_batch[:-1], speaker_batch[-1]
        added_info = {
            "game_id" : context_dict["game_id"],
            "round_index" : context_dict["round_index"],
            "speaker_context" : speaker_context,
            "listener_context" : listener_context,
        }

        return listener_batch, speaker_batch, added_info

    def getitem_joint_standard_listener(self, context_dict):
        # Generate the raw image sequence and sim idx
        context_images = [img for img in context_dict['listener_context']]
        if self.eval_scheme is None and not self.no_shuffling:
            random.shuffle(context_images)
        raw_images = process_images(self.img_dir, context_images)

        key_name = 'gt_target'
        target_idx = context_images.index(context_dict[key_name])

        sim_idx = [context_images.index(sim_img) for sim_img in context_dict["similarity_block"]]
        if len(sim_idx) < 4:
            sim_idx = sim_idx + [sim_idx[-1]]

        # Create the prompt and inputs for the listener
        target_anno = context_dict["target_anno"]
        prompt = construct_listener_full_prompt(self.processor, target_anno, target_idx, self.comprehension_prompt_type) 
        outputs = self.processor(
            text=[prompt],
            images=[raw_images],
            padding='max_length',
            max_length=self.max_listener_len,
            return_tensors="pt"
        )
        l_input_tokens = outputs['input_ids'][0, :-2]
        l_attn_mask = outputs['attention_mask'][0, :-2]
        l_attn_mask[(l_input_tokens == 0).bool()] = 0
        images = outputs['pixel_values'][0]
        l_image_attn_mask = outputs['pixel_attention_mask'][0]

        # Create the speaker inputs
        prompts = []
        for i in range(10):
            prompt = construct_speaker_full_prompt(self.processor, target_anno, i, self.generation_prompt_type)
            prompts.append(prompt)
        outputs = self.processor(
            text=prompts,
            images=[raw_images]*10,
            padding='max_length',
            max_length=self.max_speaker_len,
            return_tensors="pt"
        )
        s_input_tokens = outputs['input_ids'][:, :-1]
        s_attn_mask = outputs['attention_mask'][:, :-1]
        s_attn_mask[(s_input_tokens == 0).bool()] = 0
        s_image_attn_mask = outputs['pixel_attention_mask'][0]
        s_target_tokens = outputs['input_ids'][:, 1:]
        s_target_mask = []
        for i in range(10):
            curr_mask = self.create_speaker_caption_mask(outputs['input_ids'][i], s_attn_mask[i], i)
            s_target_mask.append(curr_mask)
        s_target_mask = torch.stack(s_target_mask, dim=0)

        return images, l_input_tokens, l_attn_mask, l_image_attn_mask, s_input_tokens, s_attn_mask, \
            s_image_attn_mask, s_target_mask, s_target_tokens, target_idx, sim_idx, context_images

    def getitem_joint_standard_speaker(self, context_dict):
        # Generate the image inputs
        context_images = [img for img in context_dict['speaker_context']]
        if self.eval_scheme is None and not self.no_shuffling: 
            random.shuffle(context_images)
        raw_images = process_images(self.img_dir, context_images)

        key_name = 'gt_target'
        target_idx = context_images.index(context_dict[key_name])

        distractors = context_dict["distractor_annos"]
        pad_size = 0
        if len(distractors) < self.max_distractor_count:
            pad_size = self.max_distractor_count - len(distractors)
            distractors = distractors + [distractors[-1] for _ in range(pad_size)]
        annotations = [context_dict['target_anno']] + distractors

        # Generate all speaker-side data
        prompts = []
        for anno in annotations:
            prompt = construct_speaker_full_prompt(self.processor, anno, target_idx, self.generation_prompt_type)
            prompts.append(prompt)
        outputs = self.processor(
            text=prompts,
            images=[raw_images]*len(annotations),
            padding='max_length',
            max_length=self.max_speaker_len,
            truncation=True,
            return_tensors="pt"
        )
        images = outputs['pixel_values'][0]
        s_input_tokens = outputs['input_ids'][:, :-1]
        s_attn_mask = outputs['attention_mask'][:, :-1]
        s_attn_mask[(s_input_tokens == 0).bool()] = 0
        s_image_attn_mask = outputs['pixel_attention_mask']
        s_target_tokens = outputs['input_ids'][:, 1:]
        s_target_mask = []
        for i in range(len(annotations)):
            curr_mask = self.create_speaker_caption_mask(outputs['input_ids'][i], s_attn_mask[i], target_idx)
            s_target_mask.append(curr_mask)
        s_target_mask = torch.stack(s_target_mask, dim=0)
        
        # Generate all listener-side data
        prompts = []
        for anno in annotations:
            prompt = construct_listener_full_prompt(self.processor, anno, target_idx, self.comprehension_prompt_type)
            prompts.append(prompt)
        outputs = self.processor(
            text=prompts,
            images=[raw_images]*len(annotations),
            padding='max_length',
            max_length=self.max_listener_len,
            truncation=True,
            return_tensors="pt"
        )
        l_input_tokens = outputs['input_ids'][:, :-2]
        l_attn_mask = outputs['attention_mask'][:, :-2]
        l_attn_mask[(l_input_tokens == 0).bool()] = 0
        l_image_attn_mask = outputs['pixel_attention_mask']

        # Generate mask for padding annotations
        annotation_mask = torch.zeros(len(annotations))
        if pad_size > 0:
            annotation_mask[-pad_size:] = 1
        annotation_mask = annotation_mask.bool()

        return images, target_idx, s_input_tokens, s_attn_mask, s_image_attn_mask, s_target_tokens, s_target_mask, \
            l_input_tokens, l_attn_mask, l_image_attn_mask, annotation_mask, context_images

    def getitem_multitask(self, idx):
        # The individual batches
        listener_batch = self.getitem_listener(idx)
        speaker_batch = self.getitem_speaker(idx)

        listener_batch, listener_info = listener_batch[:-1], listener_batch[-1]
        speaker_batch, speaker_info = speaker_batch[:-1], speaker_batch[-1]
        added_info = {
            "game_id" : listener_info["game_id"],
            "round_index" : listener_info["round_index"],
            "listener_context" : listener_info["context"],
            "speaker_context" : speaker_info["context"],
        }

        return listener_batch, speaker_batch, added_info

    def getitem_multitask_rerank(self, idx):
        # The individual batches
        listener_batch = self.getitem_listener(idx)
        speaker_batch = self.getitem_speaker_rerank(idx)

        listener_batch, listener_info = listener_batch[:-1], listener_batch[-1]
        speaker_batch, speaker_info = speaker_batch[:-1], speaker_batch[-1]
        added_info = {
            "game_id" : listener_info["game_id"],
            "round_index" : listener_info["round_index"],
            "listener_context" : listener_info["context"],
            "speaker_context" : speaker_info["context"]
        }
        return listener_batch, speaker_batch, added_info


### DATALOADER UTILITIES

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_idefics_loader(cfg, split, task, eval_scheme=None,
                       regular_evaluation=False, human_model_path="", debug_gen=False, 
                       precompute_gen=False, precompute_analysis=False):
    # Load the dataset itself
    dset = IDEFICSDataset(cfg, split, task, eval_scheme=eval_scheme,
                          regular_evaluation=regular_evaluation, human_model_path=human_model_path,
                          debug_gen=debug_gen, precompute_gen=precompute_gen,
                          precompute_analysis=precompute_analysis)
    print(f"The {split} dataset for {cfg['training_type']} has {len(dset)} examples")

    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    shuffle = split == "train" and eval_scheme is None
    bsz = cfg["batch_size"] if (split == "train" and eval_scheme is None) else cfg["test_batch_size"]
    dataloader = DataLoader(dset, batch_size=bsz, shuffle=shuffle,
                            num_workers=cfg["num_workers"], worker_init_fn=seed_worker,
                            generator=g)

    return dataloader

def get_cl_idefics_loaders(cfg, split, task):
    shuffle = True
    bsz = cfg["batch_size"]
    tr_loaders = {}

    model_family = cfg["model_family_name"]
    if cfg["use_separate_dataloaders"] or model_family in ["no_ds", "baseline"]:
        for suffix in ["listener", "speaker"]:
            real_suffix = "all" if cfg["use_separate_dataloaders"] else suffix
            g = torch.Generator()
            g.manual_seed(cfg["seed"])

            main_dset = IDEFICSDataset(cfg, split, task, suffix=real_suffix, 
                                       noise_filter=cfg[f"{suffix}_filter"])
            main_loader = DataLoader(main_dset, batch_size=bsz, shuffle=shuffle,
                                     num_workers=cfg["num_workers"], worker_init_fn=seed_worker,
                                     generator=g)
            print(suffix, len(main_loader))

            tr_loaders[f"main_{suffix}_loader"] = main_loader
            if suffix == "speaker":
                tr_loaders["main_speaker_iter"] = iter(main_loader)
                tr_loaders["main_speaker_steps"] = 0
    else:
        g = torch.Generator()
        g.manual_seed(cfg["seed"])
        suffix = "all"

        # Load the main dataset
        main_dset = IDEFICSDataset(cfg, split, task, suffix=suffix, 
                                   noise_filter=cfg["noise_filter"])
        main_loader = DataLoader(main_dset, batch_size=bsz, shuffle=shuffle,
                                 num_workers=cfg["num_workers"], worker_init_fn=seed_worker,
                                 generator=g)
        tr_loaders["main_loader"] = main_loader

    return tr_loaders

## REPEATED IDEFICS CALLS ##
def process_images(img_dir, context_images):
    raw_images = []
    for img in context_images:
        image_path = os.path.join(img_dir, f"{img}.png")
        raw_image = Image.open(image_path).convert('RGB')
        raw_images.append(raw_image)
    return raw_images

def construct_listener_full_prompt(processor, target_anno, target_idx, comprehension_prompt_type="information_after"):
    target_anno = target_anno.lower().strip()
    messages = []

    if comprehension_prompt_type == "information_after":
        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will be presented with a sequence of 10 images and a caption describing exactly one of them. "},
                    {"type" : "text", "text" : "Your task is to guess which image the caption describes. "},
                ]
            }
        )

        # User side: Images
        for i in range(10):
            if i == 0:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i}: "})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f", Image {i}: "})
            messages[0]["content"].append({"type" : "image"})

        # User side: Caption
        messages[0]["content"].append({"type" : "text", "text" : f". Caption: {target_anno}"})
        messages[0]["content"].append({"type" : "text", "text" : f" Which image does this caption describe?"})

        # Model side: Guess
        messages.append(
            {
                "role" : "assistant",
                "content" : [
                    {"type" : "text", "text" : f"The caption describes Image {target_idx}"}
                ]
            }
        )
    elif comprehension_prompt_type == "information_before":
        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will be presented with a sequence of 10 images and a caption describing exactly one of them. "},
                    {"type" : "text", "text" : "Your task is to guess which image the caption describes. "},
                    {"type" : "text", "text" : f"Your caption is: {target_anno}"}
                ]
            }
        )

        # User side: Images
        for i in range(10):
            if i == 0:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i}: "})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f", Image {i}: "})
            messages[0]["content"].append({"type" : "image"})

        # User side: Caption

        messages[0]["content"].append({"type" : "text", "text" : f" Which image does this caption describe?"})

        # Model side: Guess
        messages.append(
            {
                "role" : "assistant",
                "content" : [
                    {"type" : "text", "text" : f"The caption describes Image {target_idx}"}
                ]
            }
        )
    elif comprehension_prompt_type == "verbose_instruction":
        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will be presented with a sequence of 10 images and a caption describing exactly one of them. "},
                    {"type" : "text", "text" : "Your task is to guess which image the caption describes. "},
                ]
            }
        )

        # User side: Images
        for i in range(10):
            if i == 0:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i}: "})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f", Image {i}: "})
            messages[0]["content"].append({"type" : "image"})

        # User side: Caption
        messages[0]["content"].append({"type" : "text", "text" : f". Caption: {target_anno}"})
        messages[0]["content"].append({"type" : "text", "text" : f" Does this caption describe Image 0, 1, 2, 3, 4, 5, 6, 7, 8 or 9?"})

        # Model side: Guess
        messages.append(
            {
                "role" : "assistant",
                "content" : [
                    {"type" : "text", "text" : f"The caption describes Image {target_idx}"}
                ]
            }
        )
    elif comprehension_prompt_type == "reversed_image_description":
        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will given a sequence of 10 images and a caption describing exactly one of them. "},
                    {"type" : "text", "text" : "Your task is to guess which image the caption describes. "},
                ]
            }
        )

        # User side: Images
        for i in range(10):
            messages[0]["content"].append({"type" : "image"})
            if i == 9:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i}.\n"})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i},\n "})

        # User side: Caption
        messages[0]["content"].append({"type" : "text", "text" : f"Caption: {target_anno}"})
        messages[0]["content"].append({"type" : "text", "text" : f" Does this caption describe Image 0, 1, 2, 3, 4, 5, 6, 7, 8 or 9?"})

        # Model side: Guess
        messages.append(
            {
                "role" : "assistant",
                "content" : [
                    {"type" : "text", "text" : f"The caption describes Image {target_idx}"}
                ]
            }
        )
    elif comprehension_prompt_type == "reversed_image_description_letters":
        image_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will given a sequence of 10 images and a caption describing exactly one of them. "},
                    {"type" : "text", "text" : "Your task is to guess which image the caption describes. "},
                ]
            }
        )

        # User side: Images
        for i in range(10):
            messages[0]["content"].append({"type" : "image"})
            messages[0]["content"].append({"type" : "text", "text" : f" Image {image_labels[i]}.\n"})

        # User side: Caption
        messages[0]["content"].append({"type" : "text", "text" : f"Caption: {target_anno}"})
        messages[0]["content"].append({"type" : "text", "text" : f" Does this caption describe Image A, B, C, D, E, F, G, H, I or J?"})

        # Model side: Guess
        messages.append(
            {
                "role" : "assistant",
                "content" : [
                    {"type" : "text", "text" : f"The caption describes Image {image_labels[target_idx]}"}
                ]
            }
        )
    elif comprehension_prompt_type == "no_instruction":
        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : []
            }
        )

        # User side: Images
        for i in range(10):
            messages[0]["content"].append({"type" : "image"})
            if i == 9:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i}.\n"})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i},\n "})

        # User side: Caption
        messages[0]["content"].append({"type" : "text", "text" : f"Which image does the caption '{target_anno}' describe?" })

        # Model side: Guess
        messages.append(
            {
                "role" : "assistant",
                "content" : [
                    {"type" : "text", "text" : f"The caption describes Image {target_idx}"}
                ]
            }
        )
    else:
        print("Invalid prompt type")
        assert("False")

    return processor.apply_chat_template(messages, add_generation_prompt=False).strip()    

def construct_speaker_full_prompt(processor, target_anno, target_idx,
                                  generation_prompt_type="information_after"):
    messages = construct_speaker_base_prompt(processor, target_idx, generation_prompt_type)

    # Assistant response
    target_anno = target_anno.lower().strip()
    messages.append(
        {
            "role" : "assistant",
            "content" : [
                {"type" : "text", "text" : target_anno}
            ]
        }
    )

    return processor.apply_chat_template(messages, add_generation_prompt=False).strip()

def construct_speaker_base_prompt(processor, target_idx, generation_prompt_type="information_after", process=False):
    messages = []

    if generation_prompt_type == "information_after":
        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will be presented with a sequence of 10 images and be assigned a target image. "},
                    {"type" : "text", "text" : "Your task is to produce a caption for your target image such that anyone could guess the image from your description. "},
                ]
            }
        )

        # User side: Images
        for i in range(10):
            if i == 0:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i}: "})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f", Image {i}: "})
            messages[0]["content"].append({"type" : "image"})

        # User side: Target assignment
        messages[0]["content"].append({"type" : "text", "text" : f". Your target image is Image {target_idx}. Produce your caption now."})
    elif generation_prompt_type == "information_after_strict":
        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will be presented with a sequence of 10 images and be assigned a target image. "},
                    {"type" : "text", "text" : "Your must produce a caption that describes your target image and only your target image. "},
                    {"type" : "text", "text" : "When someone reads your description, they should be able to pick your target image from the set of 10 images. "},
                    {"type" : "text", "text" : f"Your target for this turn will be Image {target_idx}.\n"},
                ]
            }
        )

        # User side: Images
        for i in range(10):
            if i == 0:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i}: "})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f", Image {i}: "})
            messages[0]["content"].append({"type" : "image"})

        # User side: Target assignment
        messages[0]["content"].append({"type" : "text", "text" : f". Your target image is Image {target_idx}. Produce your distinctive caption now."})
    elif generation_prompt_type == "reversed_image_description":
        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will be presented with a sequence of 10 images and be assigned a target image. "},
                    {"type" : "text", "text" : "Your task is to produce a caption for your target image such that anyone could guess the image from your description.\n"},
                ]
            }
        )

        # User side: Images
        for i in range(10):
            messages[0]["content"].append({"type" : "image"})
            if i == 9:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i}.\n"})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i},\n"})

        # User side: Target assignment
        messages[0]["content"].append({"type" : "text", "text" : f"Your target image is Image {target_idx}. Produce your caption now."})
    elif generation_prompt_type == "reversed_image_description_letters":
        image_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will be presented with a sequence of 10 images and be assigned a target image. "},
                    {"type" : "text", "text" : "Your task is to produce a caption for your target image such that anyone could guess the image from your description.\n"},
                ]
            }
        )

        # User side: Images
        for i in range(10):
            if i == target_idx:
                messages[0]["content"].append({"type" : "text", "text" : f"Target image start: "})

            messages[0]["content"].append({"type" : "image"})
            messages[0]["content"].append({"type" : "text", "text" : f"Image {image_labels[i]}. "})

            if i == target_idx:
                messages[0]["content"].append({"type" : "text", "text" : f"Target image end\n"})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f"\n"})

        # User side: Target assignment
        messages[0]["content"].append({"type" : "text", "text" : f"Your target image is Image {image_labels[target_idx]}. Produce your caption now."})
    elif generation_prompt_type == "reversed_image_description_repetition":
        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will be presented with a sequence of 10 images and be assigned a target image. "},
                    {"type" : "text", "text" : "Your task is to produce a caption for your target image such that anyone could guess the image from your description.\n"},
                    {"type" : "text", "text" : f"Your target will be Image {target_idx}.\n"},
                ]
            }
        )

        # User side: Images
        for i in range(10):
            messages[0]["content"].append({"type" : "image"})
            if i == 9:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i}.\n"})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i},\n"})

        # User side: Target assignment
        messages[0]["content"].append({"type" : "text", "text" : f"Your target image is Image {target_idx}. Produce your caption now."})
    elif generation_prompt_type == "information_before":
        # User side: Intro
        messages.append(
            {
                "role" : "user",
                "content" : [
                    {"type" : "text", "text" : "You will be presented with a sequence of 10 images and be assigned a target image. "},
                    {"type" : "text", "text" : "Your task is to produce a caption for your target image such that anyone could guess the image from your description. "},
                    {"type" : "text", "text" : f"Your target image is Image {target_idx}. "},
                ]
            }
        )

        # User side: Images
        for i in range(10):
            if i == 0:
                messages[0]["content"].append({"type" : "text", "text" : f" Image {i}: "})
            else:
                messages[0]["content"].append({"type" : "text", "text" : f", Image {i}: "})
            messages[0]["content"].append({"type" : "image"})

        # User side: Target assignment
        messages[0]["content"].append({"type" : "text", "text" : f". Produce your caption now."})
    else:
        print("Invalid prompt type")
        assert("False")

    if process:
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True).strip()
        return prompt
    else:
        return messages

## GENERAL PROCESSING UTILITIES

def process_idefics_listener_generation_input(speaker_context, captions, processor, img_dir, num_samples, device):
    # First construct the prompts
    prompts, raw_images = get_listener_generation_prompts(speaker_context, captions, num_samples, img_dir, processor)

    # Process the prompts
    listener_inputs = processor(
        text=prompts,
        images=raw_images,
        padding='longest',
        return_tensors='pt'
    )

    input_tokens = listener_inputs['input_ids'][:, :-2].to(device)
    attn_mask = listener_inputs['attention_mask'][:, :-2].to(device)
    attn_mask[input_tokens == 0] = 0
    images = listener_inputs['pixel_values'].to(device)
    image_attn_mask = listener_inputs['pixel_attention_mask'].to(device)

    return input_tokens, attn_mask, images, image_attn_mask

def get_listener_generation_prompts(speaker_contexts, captions, num_samples, img_dir, processor):
    prompts = []
    all_raw_images = []

    for i, speaker_context in enumerate(speaker_contexts):
        raw_images = process_images(img_dir, speaker_context)
        for j in range(num_samples):
            curr_idx = i * num_samples + j
            caption = captions[curr_idx]
            prompt = construct_listener_full_prompt(processor, caption, 0, "verbose_instruction")

            prompts.append(prompt)
            all_raw_images.append(raw_images)
    return prompts, all_raw_images
