# File: create_prelaunch_round_data.py
# ------------------------------------
# Creates json scripts to help precompute model utterances before a deployment.

import random
import numpy as np
import os
import argparse
import pickle
import json

REFGAME_FOLDER = "game_server_side/refgame/public/games"
SPLIT_DATA_FOLDER = "/home/mog29/cogen/data_and_checkpoints/continual_learning"
TREATMENTS = ["full", "baseline", "no_ji", "no_ds", "old_full"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curr_round", type=int)
    parser.add_argument("--num_games", type=int)
    args = parser.parse_args()    
    return args

def preprocess_games(treatment_folder, treatment, num_games):
    i2t_path = os.path.join(REFGAME_FOLDER, 'idx_to_tangram.json')
    with open(i2t_path) as f:
        idx_to_tangram = json.load(f)

    # Iterate over each game json
    speaker_rounds = {}
    for json_number in range(num_games):
        # Load the game json
        game_id = f"game_json_{json_number}"
        filepath = os.path.join(treatment_folder, f"{game_id}.json")
        with open(filepath, 'r') as f:
            curr_game_data = json.load(f)
        speaker_rounds[game_id] = {}

        # Iterate over each context 
        for i, context in enumerate(curr_game_data['blocks']):
            if "anno" in context:
                continue
            if context["roles"][0] != 1:
                continue
            if context["bot_treatment"] != treatment:
                continue

            # Get the contexts
            context["img"] = [idx_to_tangram[str(img)][:-4] for img in context["img"]]
            context["tgt"] = [idx_to_tangram[str(img)][:-4] for img in context["tgt"]]
            speaker_order = context['order'][0]
            speaker_context = [0 for _ in range(10)]
            for img_idx in range(10):
                speaker_context[speaker_order[img_idx]] = context["img"][img_idx]

            listener_order = context['order'][1]
            listener_context = [context["img"][order_idx] for order_idx in listener_order]

            for j, target in enumerate(context["tgt"]):
                round_name = f"{i},{j}"
                round_dict = {
                    "speaker_context" : speaker_context,
                    "listener_context" : listener_context,
                    "chat" : "Lorem ipsum",
                    "gt_target" : target,
                    "speaker" : "placeholder",
                    "listener" : "placeholder!"
                }
                speaker_rounds[game_id][round_name] = round_dict

    return speaker_rounds

def main():
    args = get_args()

    # Iterate over each relevant round
    for treatment in TREATMENTS:
        # Get the rounds in the desired format
        treatment_folder = os.path.join(REFGAME_FOLDER, f"r{args.curr_round}_human_model")
        speaker_rounds = preprocess_games(treatment_folder, treatment, args.num_games)
        
        # Save the games
        savepath = os.path.join(SPLIT_DATA_FOLDER, f"cl_r{args.curr_round}_{treatment}_speaker_precompute.json")
        with open(savepath, 'w') as f:
            json.dump(speaker_rounds, f)

if __name__ == "__main__":
    main()
