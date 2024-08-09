"""

assumptions:
1. corpus is or fully empty or fully painted. not both.
2. each tangram is or fully empty or fully painted, not partail painted.
4. each trial has same number of tangrams
4. each context is 'unique', means that a tangram (with specific annotation) that has been choosen as 'context'
and not as 'target', will no longer be choosen as a 'context' (and therefore not as a tareget, too)


"""

import math
import os
import json
import pickle
import random
from copy import deepcopy
from re import T
from pymongo import MongoClient
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm
import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from config_generator import sample_similarity_block_base, sample_similarity_block


ROOT_PATH = "/Users/mustafaomergul/Desktop/Cornell/Research/tangrams-compgen/tangrams-ref-dev-omer"
DIVERGENCE_PATH = ROOT_PATH + "/scripts/generate_games/full_SND.json"
EMPTY_DATA_PATH =  "/Users/mustafaomergul/Desktop/Cornell/Research/kilogram/dataset/tangrams-svg"
JSON_PATH = ROOT_PATH + "/refgame/public/games"
ATTN_CHECK_PATH = ROOT_PATH + "/attention_checks"
CLIP_PATH = "/Users/mustafaomergul/Desktop/Cornell/Research/tangrams-compgen/similarity_files/clip_model0_similarities.pkl"
ROLES = [["speaker", "listener"], ["listener", "speaker"]]


def get_args():
    parser = argparse.ArgumentParser(description="Generating jsons for tangram game trials")

    # Interaction arguments
    parser.add_argument('--check_num', type=int, default=100,
                        help="Number of attention checks to generate")
    parser.add_argument('--snd_percentile', type=float, default=0.15,
                        help="Bottom percent of SND to use for targets")
    parser.add_argument('--max_similarity', type=float, default=0,
                        help="Maximum similarity distractors should have to the target")

    parser.add_argument('--sampling_method', type=str, default="sample",
                        choices=["top_k", "sample"], help="How to sample similar tangrams")
    parser.add_argument('--sampling_temperature', type=float, default=0.055,
                        help="The softmax temperature for when we sample the similarity block")
    parser.add_argument('--block_distance', type=float, default=1,
                        help="How distant should the base tangram of past blocks be to the current base?")

    # Reproducibility
    parser.add_argument('--seed', type=int,
                        help="The seed to set before dataset splitting")

    args = parser.parse_args()
    return args
    
def get_data():
    # Get the list of all paths
    paths = os.listdir(EMPTY_DATA_PATH)
    paths = [path for path in paths if ".DS_Store" not in path]
    random.shuffle(paths)

    # Remove duplicates
    for duplicate in ['page6-51.svg', 'page6-66.svg', 'page4-170.svg']:
        if duplicate in paths:
            paths.remove(duplicate)

    print(f"There are {len(paths)} tangrams total")
    return paths

def get_low_snd_data(snd_percentile):
    # Get sorted tangram-snd pairings
    with open(DIVERGENCE_PATH, 'r') as f:
        snd_map = json.load(f)
    snd_list = [(tangram, snd) for tangram, snd in snd_map.items()]
    snd_list = sorted(snd_list, key=lambda x: x[1])

    # Get the relevant amount of data
    slice_len = int(len(snd_list) * snd_percentile)
    sliced_list = snd_list[:slice_len]

    return_tangrams = [tangram for tangram, _ in sliced_list]
    return return_tangrams

def get_filtered_tangrams(similarity_scores, all_data, target, max_similarity):
    filtered_tangrams = []
    tangram_scores = similarity_scores[target[:-4]]

    for tangram, score in tangram_scores.items():
        svg_tangram = tangram + ".svg"
        if svg_tangram in all_data and score < max_similarity:
            filtered_tangrams.append(svg_tangram)

    return filtered_tangrams

def generate_attention_check(similarity_scores, tangrams_to_idx, all_data, target_candidates, args):
    context_dict = {}

    # Sample the attention check target
    target = random.sample(target_candidates, 1)[0] + ".svg"
    targets = [target]
    overall_context = [target]

    # Filter out data to exclude those too similar to the attention check
    filtered_data = get_filtered_tangrams(similarity_scores, all_data, target, args.max_similarity)

    # Get the similarity blocks
    for i in [4, 3]:
        # Sample base tangram
        base_tangram = sample_similarity_block_base(filtered_data, similarity_scores, overall_context, args)

        # Sample the similarity block
        similarity_block = sample_similarity_block(filtered_data, base_tangram, i, similarity_scores, args)
        overall_context.extend(similarity_block)

        # Filter out the corpus
        filtered_data = [tangram for tangram in filtered_data if tangram not in overall_context]

    # Get the random block
    overall_context.extend(random.sample(filtered_data, 2))

    # Get the remaining context_dict keys
    cur_roles = 0
    tangram_range = list(range(10))
    tangrams_order = [random.sample(tangram_range, 10),
                      random.sample(tangram_range, 10)]
    
    # Construct the dictionary
    context_dict = {
        "img" : [tangrams_to_idx[t] for t in overall_context],
        "tgt" : [tangrams_to_idx[t] for t in targets],
        "order" : tangrams_order,
        "roles" : ROLES[cur_roles]
    }
    
    return context_dict

if __name__ == "__main__":
    # Get the arguments for attention check generation
    args = get_args() 

    # Generate the attention checks
    attention_check_trials = []

    # Load the CLIP model for similarity
    t2idx_path = os.path.join(JSON_PATH, 'tangram_to_idx.json')
    with open(t2idx_path, 'r') as f:
        tangrams_to_idx = json.load(f)
    with open(CLIP_PATH, 'rb') as f:
        similarity_scores = pickle.load(f)
    all_data = get_data()
    target_candidates = get_low_snd_data(args.snd_percentile)

    for check_num in tqdm(range(args.check_num)):
        new_check = generate_attention_check(similarity_scores, tangrams_to_idx, all_data, target_candidates, args)
        attention_check_trials.append(new_check)

    # Save the attention checks
    path = os.path.join(ATTN_CHECK_PATH, "unannotated_complex_attention_checks.json")
    with open(path, 'w') as f:
        json.dump(attention_check_trials, f)
