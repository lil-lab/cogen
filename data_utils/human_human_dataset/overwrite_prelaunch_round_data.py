import random
import numpy as np
import os
import argparse
import pickle
import json
import torch

REFGAME_FOLDER = "game_server_side/refgame/public/games"
SPLIT_DATA_FOLDER = "/home/mog29/cogen/data_and_checkpoints/continual_learning"
TREATMENTS = ["full", "baseline", "no_ji", "no_ds", "old_full"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curr_round", type=int)
    parser.add_argument("--num_games", type=int)
    args = parser.parse_args()    
    return args

def overwrite_games(treatment_folder, treatment, num_games, round_idx):
    # Load the model predictions
    base_folder = "/home/mog29/cogen/data_and_checkpoints/experiments/joint_training"

    if treatment != "old_full":
        prediction_file = os.path.join(base_folder, f"r{round_idx-1}_{treatment}", "run",
                                       "logging", "next_round_preds_gen_preds.pth")
    else:
        prediction_file = os.path.join(base_folder, f"r0_full", "run",
                                       "logging", "next_round_preds_gen_preds.pth")
    print(treatment, prediction_file)
    preds = torch.load(prediction_file)

    # Iterate over each game json
    for json_number in range(num_games):
        # Load the game json
        game_id = f"game_json_{json_number}"
        filepath = os.path.join(treatment_folder, f"{game_id}.json")
        with open(filepath, 'r') as f:
            curr_game_data = json.load(f)

        # Iterate over each context
        for i, context in enumerate(curr_game_data['blocks']):
            if "anno" in context:
                continue
            if context["roles"][0] != 1:
                continue
            if context["bot_treatment"] != treatment:
                continue

            speaker_utterances = []
            for j, target in enumerate(context["tgt"]):
                round_name = f"{i},{j}"

                # Get the model prediction and get the utterance
                curr_preds = preds[game_id][round_name]
                if treatment in ["baseline", "no_ji"]:
                    utterance = curr_preds["utterances"]
                    speaker_utterances.append(utterance)                    
                else:
                    scores = curr_preds["joint_scores"]
                    max_idx = torch.argmax(scores).item()
                    speaker_utterances.append(curr_preds["utterances"][max_idx])

            context["bot_precomputed_utterances"] = speaker_utterances

        with open(filepath, 'w') as f:
            json.dump(curr_game_data, f)

def main():
    args = get_args()
    round_idx = args.curr_round
    num_games = args.num_games

    # Iterate over each relevant round
    for treatment in TREATMENTS:
        # Get the rounds in the desired format
        treatment_folder = os.path.join(REFGAME_FOLDER, f"r{round_idx}_human_model")
        overwrite_games(treatment_folder, treatment, num_games, round_idx)

if __name__ == "__main__":
    main()
