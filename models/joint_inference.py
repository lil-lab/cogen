# File: model_utils
# -----------------
# Contain utilities for models, such as loading and saving models

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformers import GenerationConfig
from continual_learning.train_utils import filter_targets
from data_utils.dataset import process_idefics_listener_generation_input
import pdb

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

class IdeficsJointInferenceModel(nn.Module):

    def __init__(self, listener_lambda, speaker_lambda,
                 model=None, listener=None, speaker=None):
        super().__init__()
        self.l_lambda = listener_lambda
        self.s_lambda = speaker_lambda

        self.has_shared_parameters = model is not None
        if self.has_shared_parameters:
            self.model = model
        else:
            self.listener = listener
            self.speaker = speaker

    def forward(self, inf_mode, arguments):
        if inf_mode == "joint_comprehension":
            return self.comprehension_side(arguments)
        elif inf_mode == "joint_reranking":
            return self.reranking_side(arguments)
        elif inf_mode == "comprehension":
            return self.split_comprehension_forward(arguments)
        elif inf_mode == "split_reranking":
            return self.split_reranking_forward(arguments)
        elif inf_mode == "generation":
            return self.split_generation_forward(arguments)

    def get_listener(self):
        if self.has_shared_parameters:
            return self.model
        else:
            return self.listener            

    def get_speaker(self):
        if self.has_shared_parameters:
            return self.model
        else:
            return self.speaker            

    def get_image_embeddings(self, pixel_values, pixel_attention_mask, model):
        '''
        Get image embeddings to avoid repeated computation for images during joint inference.
        Adapted from the IDEFICS-2 source code.
        '''
        # Get the model
        model = self.get_listener() if model == "listener" else self.get_speaker()
        if len(pixel_attention_mask.shape) == 5:
            pixel_attention_mask = pixel_attention_mask[:, 0].contiguous()

        # Assume images of form: BxCxcnlxHxW        
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=model.dtype)  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        # Remove padding images from the mask/pP p
        pixel_attention_mask = pixel_attention_mask.view(
            batch_size * num_images, *pixel_attention_mask.shape[2:]
        )
        pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = model.model.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_hidden_states = model.model.model.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        ).last_hidden_state

        # Modality projection & resampling
        image_hidden_states = model.model.model.connector(
            image_hidden_states, attention_mask=patch_attention_mask.view(pixel_values.size(0), -1)
        )

        return image_hidden_states

    def split_comprehension_side(self, input_tokens, attn_mask, images, image_attn_mask, index_to_token):
        '''
        Redundant with split_comprehension_forward except for the final computation. 
        Used during deployment in ray_models.py.
        '''
        listener = self.get_listener()
        all_logits = listener(
            input_ids=input_tokens,
            attention_mask=attn_mask,
            pixel_values=images, 
            pixel_attention_mask=image_attn_mask
        )['logits']
        target_logits = filter_targets(all_logits[:, -1], index_to_token)
        listener_log_probs = F.log_softmax(target_logits, dim=1)
        return listener_log_probs

    def split_comprehension_forward(self, arguments):
        input_tokens, attn_mask, images, image_attn_mask = arguments
        listener = self.get_listener()
        all_logits = listener(
            input_ids=input_tokens,
            attention_mask=attn_mask,
            pixel_values=images,
            pixel_attention_mask=image_attn_mask
        )['logits']
        return all_logits

    def split_generation_forward(self, arguments):
        input_tokens, attn_mask, images, image_attn_mask = arguments
        speaker = self.get_speaker()
        all_logits = speaker(
            input_ids=input_tokens,
            attention_mask=attn_mask,
            pixel_values=images,
            pixel_attention_mask=image_attn_mask
        )['logits']
        return all_logits

    def split_reranking_forward(self, arguments):
        images, input_tokens, attn_mask, image_attn_mask, target_tokens, target_mask = arguments

        # Get the image embeddings
        image_embeddings = self.get_image_embeddings(images, image_attn_mask, "speaker")
        embed_shape = image_embeddings.shape
        B, mult = input_tokens.shape[:2]
        C = images.shape[1]
        image_embeddings = image_embeddings.view(B, C, *embed_shape[1:])
        image_embeddings = image_embeddings.unsqueeze(1).repeat(1, mult, 1, 1, 1).view(-1, *embed_shape[1:])

        annotation_mask = torch.zeros(B, mult, device=image_embeddings.device).bool()
        _, speaker_log_probs = self.reranking_speaker_side(image_embeddings, input_tokens, attn_mask,
                                                           image_attn_mask, target_tokens, target_mask,
                                                           annotation_mask)
        return speaker_log_probs

    def comprehension_side(self, arguments):
        images, l_input_tokens, l_attn_mask, l_image_attn_mask, index_to_token, \
            s_input_tokens, s_attn_mask, s_image_attn_mask, s_target_mask, s_target_label = arguments

        if self.has_shared_parameters:
            image_embeddings = self.get_image_embeddings(images, l_image_attn_mask, "listener")
            listener_log_probs = self.comprehension_listener_side(
                image_embeddings, l_input_tokens, l_attn_mask, l_image_attn_mask, index_to_token
            ) # TODO

            speaker_log_probs = self.comprehension_speaker_side(
                image_embeddings, s_input_tokens, s_attn_mask, s_image_attn_mask, s_target_mask, s_target_label
            )
        else:
            # Deprecated and not used in experiments
            listener_embeddings = self.get_image_embeddings(images, l_image_attn_mask, "listener")
            listener_log_probs = self.comprehension_listener_side(
                listener_embeddings, l_input_tokens, l_attn_mask, l_image_attn_mask, index_to_token
            )        

            speaker_embeddings = self.get_image_embeddings(images, "speaker")
            speaker_log_probs = self.comprehension_speaker_side(
                speaker_embeddings, s_input_tokens, s_attn_mask, s_image_attn_mask, s_target_mask, s_target_label
            )

        joint_log_probs = self.comprehension_reranking(listener_log_probs, speaker_log_probs)
        return listener_log_probs, speaker_log_probs, joint_log_probs

    def comprehension_listener_side(self, image_encoder_embeddings, input_tokens, attn_mask, image_attn_mask,
                                    index_to_token):
        listener = self.get_listener()
        all_logits = listener(
            input_ids=input_tokens,
            attention_mask=attn_mask,
            image_hidden_states=image_encoder_embeddings,
            pixel_attention_mask=image_attn_mask
        )['logits']

        target_logits = filter_targets(all_logits[:, -1], index_to_token) # BxC
        listener_log_probs = F.log_softmax(target_logits, dim=1)
        return listener_log_probs

    def comprehension_speaker_side(self, image_encoder_embeddings, input_tokens, attn_mask, image_attn_mask,
                                   target_mask, target_label):
        # Expand embeddings
        B, C = input_tokens.shape[:2]
        embed_shape = image_encoder_embeddings.shape
        image_encoder_embeddings = image_encoder_embeddings.view(B, C, *embed_shape[1:])
        image_encoder_embeddings = image_encoder_embeddings.unsqueeze(1).repeat(1, C, 1, 1, 1).view(-1, *embed_shape[1:])
        input_tokens = input_tokens.view(B*C, -1)
        attn_mask = attn_mask.view(B*C, -1)

        # Forward pass
        speaker = self.get_speaker()
        all_logits = speaker(
            input_ids=input_tokens,
            attention_mask=attn_mask,
            image_hidden_states=image_encoder_embeddings,
        )['logits']
        
        # Get tokenwise probabilities
        all_log_probs = F.log_softmax(all_logits, dim=2)
        target_label = target_label.view(B*C, -1).unsqueeze(2)
        target_mask = target_mask.view(B*C, -1)
        token_log_probs = torch.gather(all_log_probs, 2, target_label).squeeze(2) # BCxT

        # Compute the log probabilities
        token_log_probs = token_log_probs * target_mask
        utterance_log_probs = torch.sum(token_log_probs, dim=1).view(B, C)

        return utterance_log_probs

    def comprehension_reranking(self, listener_log_probs, speaker_log_probs):
        rerank_weights = self.l_lambda * listener_log_probs + (1 - self.l_lambda) * speaker_log_probs
        rerank_denominator = torch.logsumexp(rerank_weights, dim=1).unsqueeze(1)
        rerank_log_distribution = rerank_weights - rerank_denominator            
        return rerank_log_distribution
    
    def reranking_side(self, arguments):
        images, label, s_input_tokens, s_attn_mask, s_image_attn_mask, s_target_tokens, s_target_mask, \
            l_input_tokens, l_attn_mask, l_image_attn_mask, \
            index_to_token, annotation_mask = arguments

        # Repeat image embeddings according to number of distractors
        if self.has_shared_parameters:
            image_embeddings = self.get_image_embeddings(images, s_image_attn_mask, "speaker")
            embed_shape = image_embeddings.shape
            B, mult = s_input_tokens.shape[:2]
            C = images.shape[1]
            image_embeddings = image_embeddings.view(B, C, *embed_shape[1:])
            image_embeddings = image_embeddings.unsqueeze(1).repeat(1, mult, 1, 1, 1).view(-1, *embed_shape[1:])

            speaker_logits, speaker_log_probs = self.reranking_speaker_side(image_embeddings, s_input_tokens,
                                                                            s_attn_mask, s_image_attn_mask,
                                                                            s_target_tokens, s_target_mask,
                                                                            annotation_mask)

            listener_log_probs = self.reranking_listener_side(image_embeddings, l_input_tokens, l_attn_mask,
                                                              l_image_attn_mask, label, index_to_token,
                                                              annotation_mask)
        else:
            # Deprecated and no longer used in main experiments
            image_embeddings = self.get_image_embeddings(images, s_image_attn_mask, "speaker")
            embed_shape = image_embeddings.shape
            B, mult = s_input_tokens.shape[:2]
            C = images.shape[1]
            image_embeddings = image_embeddings.view(B, C, *embed_shape[1:])
            image_embeddings = image_embeddings.unsqueeze(1).repeat(1, mult, 1, 1, 1).view(-1, *embed_shape[1:])

            speaker_logits, speaker_log_probs = self.reranking_speaker_side(image_embeddings, s_input_tokens,
                                                                            s_attn_mask, s_image_attn_mask,
                                                                            s_target_tokens, s_target_mask,
                                                                            annotation_mask)


            image_embeddings = self.get_image_embeddings(images, l_image_attn_mask, "listener")
            embed_shape = image_embeddings.shape
            B, mult = s_input_tokens.shape[:2]
            C = images.shape[1]
            image_embeddings = image_embeddings.view(B, C, *embed_shape[1:])
            image_embeddings = image_embeddings.unsqueeze(1).repeat(1, mult, 1, 1, 1).view(-1, *embed_shape[1:])

            listener_log_probs = self.reranking_listener_side(image_embeddings, l_input_tokens, l_attn_mask,
                                                              l_image_attn_mask, label, index_to_token, annotation_mask)

        # Full forward passes
        utterance_distribution = self.reranking_combination(speaker_log_probs, listener_log_probs)
        return speaker_logits, speaker_log_probs, listener_log_probs, utterance_distribution
        

    def reranking_speaker_side(self, image_embeddings, input_tokens, attn_mask, image_attn_mask,
                               target_tokens, target_mask, annotation_mask):
        # Flatten inputs and outputs
        B, mult = input_tokens.shape[:2]
        input_tokens = input_tokens.view(B*mult, -1)
        attn_mask = attn_mask.view(B*mult, -1)
        target_tokens = target_tokens.view(B*mult, -1).unsqueeze(-1)
        target_mask = target_mask.view(B*mult, -1)

        # Forward pass: Compute utterance probabilities for all
        speaker = self.get_speaker()
        all_logits = speaker(
            input_ids=input_tokens,
            attention_mask=attn_mask,
            image_hidden_states=image_embeddings,
        )['logits']
        
        # Compute utterance log probabilities
        all_log_probs = F.log_softmax(all_logits, dim=2)
        token_log_probs = torch.gather(all_log_probs, 2, target_tokens).squeeze(2) # BCxT
        token_log_probs = token_log_probs * target_mask
        utterance_log_probs = torch.sum(token_log_probs, dim=1).view(B, mult)
        utterance_log_probs[annotation_mask] = float('-inf') # Mask in the event there aren't 9 distractors
            
        return all_logits, utterance_log_probs

    def reranking_listener_side(self, image_embeddings, input_tokens, attn_mask, image_attn_mask,
                                label, index_to_token, annotation_mask):
        # Flatten inputs and outputs
        B, mult = input_tokens.shape[:2]
        input_tokens = input_tokens.view(B*mult, -1)
        attn_mask = attn_mask.view(B*mult, -1)
        label = label.unsqueeze(1).repeat(1, mult).view(-1).unsqueeze(1)

        # Forward pass: Compute listener log-probs
        listener = self.get_listener()
        all_logits = listener(
            input_ids=input_tokens,
            attention_mask=attn_mask,
            image_hidden_states=image_embeddings,
        )['logits']

        target_logits = filter_targets(all_logits[:, -1], index_to_token) # BmultxC
        listener_log_probs = F.log_softmax(target_logits, dim=1) #BmultxC
        utterance_log_probs = torch.gather(listener_log_probs, 1, label).squeeze(1).view(B, mult)

        utterance_log_probs[annotation_mask] = float('-inf') # Mask in the event there aren't mult distractors

        return utterance_log_probs

    def reranking_combination(self, speaker_utterance_log_probs, listener_utterance_log_probs):
        weights = self.s_lambda * speaker_utterance_log_probs + (1-self.s_lambda) * listener_utterance_log_probs
        rerank_denominator = torch.logsumexp(weights, dim=1).unsqueeze(1)
        rerank_log_distribution = weights - rerank_denominator
        return rerank_log_distribution

    def split_generate(self, input_tokens, attn_mask, images, image_attn_mask, processor,
                       max_steps=25, sampling_type="nucleus", temperature=1.0,
                       top_k=40, top_p=0.9, repetition_penalty=1, num_samples=1):
        # (1) Perform generation
        speaker = self.get_speaker()
        generation_config = GenerationConfig(
            max_new_tokens=max_steps,
            do_sample=True,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_samples,
            output_hidden_states=True,
            return_dict_in_generate=True
        ) 
        outputs = speaker.generate(
            input_ids=input_tokens,
            attention_mask=attn_mask,
            pixel_values=images,
            pixel_attention_mask=image_attn_mask,
            generation_config=generation_config,
            use_cache=True
        )

        # (2) Get the speaker captions
        B = input_tokens.shape[0]
        observed_steps = len(outputs['hidden_states'])
        filtered_seqs = []
        for seq in outputs['sequences']:
            filtered_seqs.append(seq[-observed_steps:])
        speaker_outputs = processor.batch_decode(filtered_seqs, skip_special_tokens=True)
        
        # (3) Get the speaker log probabilities
        target_outputs = torch.stack(filtered_seqs, dim=0) # BNxT
        target_mask = target_outputs != 0
        final_states = torch.stack([outputs['hidden_states'][i][-1][:, -1] for i in range(observed_steps)], dim=1) # BNxTxD
        token_logits = speaker.lm_head(final_states) # BNxTxV
        token_log_probs = F.log_softmax(token_logits, dim=2)
        token_log_probs = torch.gather(token_log_probs, 2, target_outputs.unsqueeze(2)).squeeze(2)
        
        # (4) Choose the output with the top probability
        if B == 1:
            utterance_log_probs = torch.sum(token_log_probs * target_mask, dim=1).view(num_samples) # N
            best_idx = torch.argmax(utterance_log_probs).item()
            return [speaker_outputs[best_idx]]
        else:
            utterance_log_probs = torch.sum(token_log_probs * target_mask, dim=1).view(B, num_samples) # N
            best_indices = torch.argmax(utterance_log_probs, dim=1)
            choices = []
            for i in range(B):
                curr_index = num_samples * i + best_indices[i].item()                
                choices.append(speaker_outputs[curr_index])
            return choices
            

    def generate(self, images, s_input_tokens, s_attn_mask, s_image_attn_mask, label,
                 image_paths, processor, image_dir, index_to_token,
                 max_steps=25, sampling_type="nucleus", temperature=1.0, top_k=40,
                 top_p=0.9, repetition_penalty=1, num_samples=10):
        # Get the repeated image embeddings; assume parameter sharing
        image_embeddings = self.get_image_embeddings(images, s_image_attn_mask, "speaker") 

        # Sample utterances from the speaker
        speaker_utterance_log_probs, speaker_utterances = self.generate_speaker_side(processor, images, s_input_tokens,
                                                                                     s_attn_mask, s_image_attn_mask, max_steps,
                                                                                     sampling_type, temperature,
                                                                                     top_k, top_p, repetition_penalty,
                                                                                     num_samples) # BxN, BN list

        # Get probabilities for the utterances from the listener
        listener_log_probs = self.generate_listener_side(image_embeddings, speaker_utterances, label, image_paths, processor,
                                                         image_dir, index_to_token, num_samples)

        # Reranked selection
        utterance_weights = self.s_lambda*speaker_utterance_log_probs + (1-self.s_lambda)*listener_log_probs
        chosen_indices = torch.argmax(utterance_weights, dim=1)
        choices = []
        for i in range(speaker_utterance_log_probs.shape[0]):
            curr_index = num_samples * i + chosen_indices[i].item()
            choices.append(speaker_utterances[curr_index])
            
        return choices, speaker_utterances, listener_log_probs, speaker_utterance_log_probs, utterance_weights

    def generate_speaker_side(self, processor, images, s_input_tokens, s_attn_mask, s_image_attn_mask, max_steps,
                              sampling_type, temperature, top_k, top_p, repetition_penalty, num_samples):
        # (1) Perform generation
        speaker = self.get_speaker()
        generation_config = GenerationConfig(
            max_new_tokens=max_steps,
            do_sample=True,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_samples,
            output_hidden_states=True,
            return_dict_in_generate=True
        ) 
        outputs = speaker.generate(
            input_ids=s_input_tokens,
            attention_mask=s_attn_mask,
            pixel_values=images,
            pixel_attention_mask=s_image_attn_mask,
            generation_config=generation_config,
            use_cache=True
        )

        # (2) Get the speaker captions
        B = s_input_tokens.shape[0]
        observed_steps = len(outputs['hidden_states'])
        filtered_seqs = []
        for seq in outputs['sequences']:
            filtered_seqs.append(seq[-observed_steps:])
        speaker_outputs = processor.batch_decode(filtered_seqs, skip_special_tokens=True)

        # (3) Get the speaker log probabilities
        target_outputs = torch.stack(filtered_seqs, dim=0) # BNxT
        target_mask = target_outputs != 0
        final_states = torch.stack([outputs['hidden_states'][i][-1][:, -1] for i in range(observed_steps)], dim=1) # BNxTxD
        token_logits = speaker.lm_head(final_states) # BNxTxV
        token_log_probs = F.log_softmax(token_logits, dim=2)
        token_log_probs = torch.gather(token_log_probs, 2, target_outputs.unsqueeze(2)).squeeze(2)
        utterance_log_probs = torch.sum(token_log_probs * target_mask, dim=1).view(B, num_samples) # BxN
            
        return utterance_log_probs, speaker_outputs

    def generate_listener_side(self, image_embeddings, speaker_utterances, label, image_paths, processor,
                               image_dir, index_to_token, num_samples):
        # Construct the inputs
        B = label.shape[0]
        embed_shape = image_embeddings.shape
        image_embeddings = image_embeddings.view(B, -1, *embed_shape[1:])
        image_embeddings = image_embeddings.unsqueeze(1).repeat(1, num_samples, 1, 1, 1).view(-1, *embed_shape[1:])

        l_batch = process_idefics_listener_generation_input(image_paths, speaker_utterances, processor, 
                                                            image_dir, num_samples, image_embeddings.device)
        l_input_tokens, l_attn_mask, _, l_image_attn_mask = l_batch
        label = label.unsqueeze(1).repeat(1, num_samples).view(-1).unsqueeze(1)

        # Forward pass
        listener = self.get_listener()
        all_logits = listener(
            input_ids=l_input_tokens,
            attention_mask=l_attn_mask,
            image_hidden_states=image_embeddings,
            pixel_attention_mask=l_image_attn_mask
        )['logits']

        target_logits = filter_targets(all_logits[:, -1], index_to_token)
        listener_log_probs = F.log_softmax(target_logits, dim=1)
        utterance_log_probs = torch.gather(listener_log_probs, 1, label).squeeze(1).view(B, num_samples)

        return utterance_log_probs
        

