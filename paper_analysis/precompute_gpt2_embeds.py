import numpy as np
import torch
import os
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from evaluate import load
from mauve.compute_mauve import get_features_from_input
import argparse

SPLIT_FOLDER = '/home/mog29/cogen/data_and_checkpoints/continual_learning/analysis'
ANALYSIS_CACHE = '/home/mog29/cogen/paper_analysis/analysis_cache/gpt2_features'

def get_utterances(treatment, deployment_round):
    if treatment == "human":
        gen_path = os.path.join(SPLIT_FOLDER, f'human_human.json')
    else:
        gen_path = os.path.join(SPLIT_FOLDER, f'r{deployment_round}_{treatment}.json')
    with open(gen_path, 'r') as f:
        data = json.load(f)

    pairs = []
    utterances = []

    for game_id, game_dict in data.items():
        if f'r{deployment_round}' not in game_id:
            continue
        for round_idx, round_dict in game_dict.items():
            pairs.append((game_id, round_idx))
            utterances.append(round_dict['chat'].lower())

    return pairs, utterances

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treatment_name', type=str)
    parser.add_argument('--round_idx', type=int)
    args = parser.parse_args()
    return args

def main():
    args = get_args() 
    treatment = args.treatment_name
    deployment_round = args.round_idx

    # Get features
    pairs, utterances = get_utterances(treatment, deployment_round)
    features = get_features_from_input(
        None, None, utterances, "gpt2-large", 1024, 0,
        name="p", verbose=True, batch_size=64, use_float64=False,
    )
    print("Extracted features: ", type(features), features.shape)

    # Map features to (game_id, round_idx) pairs
    feature_dict = {}
    for i, (game_id, round_idx) in enumerate(pairs):
        feature = features[i]
        if game_id not in feature_dict:
            feature_dict[game_id] = {}
        feature_dict[game_id][round_idx] = feature

    # Save the precomputed feature dict
    savepath = os.path.join(ANALYSIS_CACHE, f'r{deployment_round}_{treatment}.pth')
    torch.save(feature_dict, savepath)

if __name__ == "__main__":
    main()

