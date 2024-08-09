# File: model_manager
# -------------------
# Script for making predictions for listeners and speakers
# in an asynchronous manner

import random
import os
import json
from abc import ABC, abstractmethod
import torch
from PIL import Image
import asyncio
from bots.ray_models import RayModel

class BaseManager(ABC):

    @abstractmethod
    def listener_predict(self, image_paths, description, target_path):
        '''
        The interface assumes the following as input:
            * image_paths: A list of dictionaries where the tangram path is mapped
                           to the key "path"
            * description: A string representing the description for the current target
            * target_path: Not to be used outside of debugging. The path of the target image
        The interface should produce an image path (str) as an output
        '''
        pass

    @abstractmethod
    def speaker_predict(self, image_paths, target_path, attn_check_anno):
        '''
        The interface assumes the following as input:
            * image_paths: A list of dictionaries where the tangram path is mapped
                           to the key "path"
            * target_path: The path of the target image
            * attn_check_anno: A precomputed utterance, often for the attention check, that the
                               class should return rather than predicting.
        The interface should produce a description (str) as output
        '''
        pass
        
class DummyManager(BaseManager):

    async def listener_predict(self, image_paths, description, target_path):
        image_paths = [path["path"] for path in image_paths]
        random_image = random.choice(image_paths)
        return random_image

    async def speaker_predict(self, image_paths, target_path, attn_check_anno):
        if attn_check_anno == "":
            return "Hi! The target is a dog!!!"
        else:
            return attn_check_anno

class OracleManager(BaseManager):

    def __init__(self, config):
        # Get path for all annotations
        data_dir = config["data_dir"]
        json_path = os.path.join(data_dir, 'full.json')
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        
    async def listener_predict(self, image_paths, description, target_path):
        return target_path

    async def speaker_predict(self, image_paths, target_path, attn_check_anno):
        # Sample a random annotation for the target
        target_name = target_path.split('.')[0]
        target_annos = self.annotations[target_name]['annotations']
        sampled_anno = random.choice(target_annos)['whole']['wholeAnnotation']
        return f"Hi! I am a non-pragmatic oracle. The target annotation is: {sampled_anno}"

class ModelManager(BaseManager):

    def __init__(self, config):
        # Load the models
        self.num_models = config["num_duplicates"]
        self.models = [RayModel.remote(config) for _ in range(self.num_models)]

    async def listener_predict(self, image_paths, description, target_path):
        curr_idx = random.randint(0, self.num_models-1)
        return await self.models[curr_idx].listener_predict.remote(image_paths, description)

    async def speaker_predict(self, image_paths, target_path, attn_check_anno):
        if attn_check_anno == "":
            curr_idx = random.randint(0, self.num_models-1)
            return await self.models[curr_idx].speaker_predict.remote(image_paths, target_path)
        else:
            return attn_check_anno

def construct_manager(config):
    if config["manager_type"] == "dummy":
        return DummyManager()
    elif config["manager_type"] == "oracle":
        return OracleManager(config)
    elif config["manager_type"] == "model":
        return ModelManager(config)
