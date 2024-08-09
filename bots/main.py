# File: main
# ----------
# Main event manager for serving bots to Empirica.

import tornado.ioloop
import tornado.web
import asyncio
from time import time
import argparse
import json
import os
import numpy as np

from bots.model_manager import construct_manager
import yaml

import ray

class BotTaskAllocator():

    def __init__(self, configs):
        # Initialize the model managers for each config
        self.model_managers = {}
        for config in configs:
            self.model_managers[config["treatment_name"]] = construct_manager(config)

    async def respond_to_listener_request(self, bot_treatment, image_paths, description, target_path):
        return await self.model_managers[bot_treatment].listener_predict(image_paths, description, target_path)

    async def respond_to_speaker_request(self, bot_treatment, image_paths, target_path, attn_check_anno):
        return await self.model_managers[bot_treatment].speaker_predict(image_paths, target_path, attn_check_anno)

class ListenerServer(tornado.web.RequestHandler):

    async def post(self):
        start_time = time()

        print("In ListenerServer")
        args_bytes = self.request.body
        args_str = args_bytes.decode("utf-8") 
        args = json.loads(args_str)

        image_paths = args["image_paths"]
        description = args["description"]
        target = args['target']
        bot_treatment = args["bot_treatment"]
        round_id = args['round_id']
        game_id = args['game_id']

        # build request and ask the model
        print(f"asking the {bot_treatment} model...")
        print(f"Description at game_id {game_id} round_id {round_id}: {description}")

        target_path = await bot_task_allocator.respond_to_listener_request(bot_treatment, image_paths,
                                                                           description, target)
        time_passed = time() - start_time
        print(f"Predicted target path: {target_path}.")
        print(f"Took {time_passed} seconds")

        # Sleep to avoid immediately responding
        time_remaining = 4 - time_passed
        sleep_time = max(np.random.normal(time_remaining, 0.5), 0)
        if time_remaining > 0 and sleep_time > 0:
            await asyncio.sleep(sleep_time)

        self.write({"path" : target_path, "timePassed" : time_passed, "gameId" : game_id, "roundId" : round_id})

        print("end prediction\n")

class SpeakerServer(tornado.web.RequestHandler):
    
    async def post(self):
        start_time = time()

        print("In SpeakerServer")
        args_bytes = self.request.body
        args_str = args_bytes.decode("utf-8")
        args = json.loads(args_str)

        image_paths = args["image_paths"]
        target_path = args["target"]
        bot_treatment = args["bot_treatment"]
        attn_check_anno = args['attnCheckAnno']
        round_id = args['round_id']
        game_id = args['game_id']

        # build request and ask the model
        print(f"asking the {bot_treatment} model...")
        print(f'Target path at game_id {game_id} round_id {round_id}: {target_path}')

        description = await bot_task_allocator.respond_to_speaker_request(bot_treatment, image_paths,
                                                                          target_path, attn_check_anno)
        time_passed = time() - start_time

        print(f"Predicted description: {description}.")
        print(f"Took {time_passed} seconds")

        # Sleep to avoid immediately responding
        time_remaining = 5 - time_passed
        sleep_time = max(np.random.normal(time_remaining, 0.5), 0)
        if time_remaining > 0 and sleep_time > 0:
            await asyncio.sleep(sleep_time)

        self.write({"description" : description, "timePassed" : time_passed, "gameId" : game_id, "roundId" : round_id})
        print("end generation\n")

def get_args():
    parser = argparse.ArgumentParser(description="Serving bots to Empirica")
    parser.add_argument('--exp_config_dir', type=str,
                        default="configs/bot_configs/experiment_configs")
    parser.add_argument('--manager_config_dir', type=str,
                        default="configs/bot_configs/manager_configs")
    parser.add_argument('--configuration_file', type=str,
                        help="A configuration file specifying the models to launch for this experiment")
    parser.add_argument('--launch_system', type=str, choices=["g2", "aws"], default='g2',
                        help="What system are we launching experiments on?")
    parser.add_argument('--num_gpus', type=int, default=1)

    args = parser.parse_args()
    return args

def load_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def read_experiment_configs(exp_args):
    yaml_path = os.path.join(exp_args.exp_config_dir, exp_args.configuration_file)
    yaml_file = load_yaml(yaml_path)
    data_prefix = "/home/mog29" if exp_args.launch_system == "g2" else "/home/ubuntu"

    configs = []
    for cfg in yaml_file["configs"]:
        curr_config = load_yaml(os.path.join(exp_args.manager_config_dir, cfg))
        curr_config["data_prefix"] = data_prefix
        configs.append(curr_config)

    return configs

def make_app():
    app = tornado.web.Application([
        (r"/generate_description", SpeakerServer),
        (r"/predict_target", ListenerServer),
    ])

    return app

async def main():
    app = make_app()
    app.listen(8080)
    shutdown_event = asyncio.Event()

    print("Beginning wait")

    await shutdown_event.wait()

if __name__ == "__main__":
    # Get the experiment arguments
    experiment_args = get_args()
    ray.init(num_gpus=experiment_args.num_gpus)
    configs = read_experiment_configs(experiment_args)

    bot_task_allocator = BotTaskAllocator(configs)
    asyncio.run(main())
