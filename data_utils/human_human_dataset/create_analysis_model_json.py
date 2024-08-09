import random
import numpy as np
import os
import argparse
import pickle
import json
from tqdm import tqdm
import torch

REFGAME_FOLDER = "/home/cogen/game_server_side/refgame"
SPLIT_DATA_FOLDER = "/home/mog29/cogen/data_and_checkpoints/continual_learning"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round_idx", type=int)
    parser.add_argument("--treatment", type=str)
    parser.add_argument("--sample_suffix", type=str, default="")
    args = parser.parse_args()    
    return args

def get_model_preds(args):
    round_idx = args.round_idx - 1
    treatment = args.treatment
    base_folder = "/home/mog29/cogen/data_and_checkpoints/experiments/joint_training"
    pth_name = "analysis_preds_gen_preds.pth" if args.sample_suffix == "" else f"analysis_preds_{args.sample_suffix}_gen_preds.pth"
    prediction_file = os.path.join(base_folder, f"r{round_idx}_{treatment}", "run", "logging", pth_name)
    return torch.load(prediction_file)

def save_games(args, model_preds):
    # Load the human-human base json
    basepath = os.path.join("/home/mog29/cogen/data_and_checkpoints/continual_learning",
                            "analysis", "human_human.json")
    with open(basepath, 'r') as f:
        standard_data = json.load(f)

    # Iterate over model predictions to overwrite things
    treatment = args.treatment
    deployment_round_idx = args.round_idx

    for game_id, game_dict in tqdm(standard_data.items()):
        for round_idx, round_dict in game_dict.items():
            curr_preds = model_preds[game_id][round_idx]
            if treatment in ["baseline", "no_ji"]:
                utterance = curr_preds["utterances"]
            else:
                scores = curr_preds["joint_scores"]
                max_idx = torch.argmax(scores).item()
                utterance = curr_preds["utterances"][max_idx]

            round_dict["chat"] = utterance

    if args.sample_suffix == "":
        savepath = os.path.join("/home/mog29/cogen/data_and_checkpoints/continual_learning",
                                "analysis", f"r{deployment_round_idx}_{treatment}.json")
    else:
        savepath = os.path.join("/home/mog29/cogen/data_and_checkpoints/continual_learning",
                                "analysis", f"r{deployment_round_idx}_{treatment}_{args.sample_suffix}.json")
    with open(savepath, 'w') as f:
        json.dump(standard_data, f)

def main():
    args = get_args()
    model_preds = get_model_preds(args)
    save_games(args, model_preds)

if __name__ == "__main__":
    main()
