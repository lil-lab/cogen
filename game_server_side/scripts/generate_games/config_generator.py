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
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import torch
import torch.nn.functional as F
import nltk
from tqdm import tqdm

ROOT_PATH = "/home/mog29/clean_compgen_repos/cleaned-tangrams-compgen/tangrams-ref-dev-omer"
CLIP_PATH = "/home/mog29/clean_compgen_repos/cleaned-tangrams-compgen/similarity_files/clip_model0_similarities.pkl"
EMPTY_DATA_PATH =  "/home/mog29/compgen_saved_files/kilogram/dataset/tangrams-svg"
JSON_PATH = ROOT_PATH + "/refgame/public/games"

COMPLEX_ATTN_CHECK_PATH = ROOT_PATH + "/attention_checks/complex_attention_checks.pkl"
with open(COMPLEX_ATTN_CHECK_PATH, 'rb') as f:
    COMPLEX_ALL_ATTN_CHECKS = pickle.load(f)
ROLES = [[0, 1], [1, 0]]


def get_args():
    parser = argparse.ArgumentParser(description="Generating jsons for tangram game trials")

    # Interaction arguments
    parser.add_argument('--interaction_type', type=str, choices=["hh", "hm"],
                        help="What types of interactions will we generate?")
    parser.add_argument('--final_round', action='store_true',
                        help="If we will generate for the final round (50 rather than 40)")
    parser.add_argument('--human_model_type', type=str, default="", choices=["", "r1", "r2+"],
                        help="If we are generating human-model games, what round will our ablations contain?")
    parser.add_argument('--num_contexts', type=int, default=200,
                        help="Number of contexts to generate per underlying system (for individual listener or speaker)")
    parser.add_argument('--block_size', type=int, default=10,
                        help="Number of tangrams per block")
    parser.add_argument('--targets_per_block', type=int, default=10,
                        help="Number of target tangrams per block")
    parser.add_argument('--restricted_data_file', type=str, default="",
                        help="If set, contains a file containing a restricted dataset to generate examples from")

    # Pragmatic context arguments
    parser.add_argument('--similarity_blocks_per_block', type=int, default=3,
                        help="Number of similarity blocks in a given context block")
    parser.add_argument('--sampling_method', type=str, default="sample",
                        choices=["top_k", "sample"], help="How to sample similar tangrams")
    parser.add_argument('--sampling_temperature', type=float, default=0.055,
                        help="The softmax temperature for when we sample the similarity block")
    parser.add_argument('--block_distance', type=float, default=1,
                        help="How distant should the base tangram of past blocks be to the current base?")

    # Saving arguments
    parser.add_argument('--subfolder_name', type=str,
                        help="The name of the subfolder to place the jsons into")
    parser.add_argument('--comment', type=str, default="No comment.",
                        help="Comment associated with the json batch")

    # Reproducibility
    parser.add_argument('--seed', type=int,
                        help="The seed to set before dataset splitting")

    args = parser.parse_args()
    return args
    
def get_all_complete_game_blocks(curr_corpus, tangrams_to_idx, clip_model, args):
    if args.interaction_type == "hm":
        if args.final_round:
            treatment_names = ["full", "no_ji", "no_ds", "baseline", "old_full"]
        else:
            treatment_names = ["full", "no_ji"] if args.human_model_type == "r1" else ["full", "no_ji", "no_ds", "baseline"]
    else:
        treatment_names = ["human"]
    block_dictionary = {
        treatment : {
            "4_blocks" : [],
            "3_speaker" : [],
            "3_listener" : [],
        } for treatment in treatment_names
    }

    for treatment in treatment_names:
        for game_num in tqdm(range(args.num_contexts)):
            listener_4s, listener_3s = generate_complete_game(curr_corpus, tangrams_to_idx, clip_model, args,
                                                              treatment, game_num, 0)
            block_dictionary[treatment]["4_blocks"].extend(listener_4s)
            block_dictionary[treatment]["3_listener"].extend(listener_3s)

            speaker_4s, speaker_3s = generate_complete_game(curr_corpus, tangrams_to_idx, clip_model, args,
                                                            treatment, game_num, 1)
            block_dictionary[treatment]["4_blocks"].extend(speaker_4s)
            block_dictionary[treatment]["3_speaker"].extend(speaker_3s)


    return block_dictionary

def generate_complete_game(curr_corpus, tangrams_to_idx, clip_model, args, treatment, context_num, cur_roles):
    # First generate the context dict
    context_dict = get_pragmatic_context(curr_corpus, tangrams_to_idx, clip_model, args)
    block_size = 10
    tangram_range = list(range(block_size))
    tangrams_order = [random.sample(tangram_range, block_size),
                      random.sample(tangram_range, block_size)]
    context_dict['order'] = tangrams_order
    context_dict["roles"] = ROLES[cur_roles]
    
    # Next: Split the context dict into 3 groups of targets
    four_blocks = []
    three_blocks = []

    random.shuffle(context_dict["tgt"])
    for i in range(3):
        start = i*3
        end = (i+1)*3 if i < 2 else 10

        split_context_dict = {
            "img" : context_dict["img"],
            "similarity_blocks" : context_dict["similarity_blocks"],
            "tgt" : context_dict["tgt"][start:end],
            "order" : context_dict["order"],
            "roles" : context_dict["roles"],
            "id" : (context_num, cur_roles),
            "bot_treatment" : treatment
        }

        if (end - start) == 3:
            three_blocks.append(split_context_dict)
        else:
            four_blocks.append(split_context_dict)

    return four_blocks, three_blocks

def regroup_complete_game_blocks(block_dictionary, tangrams_to_idx, args):
    # First, determine the context order
    if args.final_round and args.interaction_type == "hm":
        treatment_order = ["full", "no_ji", "no_ds", "baseline", "old_full"]
        total_games = args.num_contexts * 2
    elif args.final_round and args.interaction_type == "hh":
        treatment_order = ["human"] * 5
        total_games = args.num_contexts * 2 // 5
    else:
        if args.interaction_type == "hm" and args.human_model_type == "r1":
            treatment_order = ["full", "full", "no_ji", "no_ji"]
            total_games = args.num_contexts
        elif args.interaction_type == "hm" and args.human_model_type != "r1":
            treatment_order = ["full", "no_ji", "no_ds", "baseline"]
            total_games = args.num_contexts * 2
        else:
            treatment_order = ["human"] * 4
            total_games = args.num_contexts // 2

    # Next, shuffle the context dicts for each system
    for treatment in treatment_order:
        for block_key in ["4_blocks", "3_speaker", "3_listener"]:
            random.shuffle(block_dictionary[treatment][block_key])

    # Create games containing 40 (or 50) targets
    all_jsons = []
    for i in range(total_games):
        four_blocks = []
        three_listener_blocks = []
        three_speaker_blocks = []

        # Get the blocks for each participating system
        for treatment in treatment_order:
            for block_key in ["4_blocks", "3_speaker", "3_listener"]:
                if block_key == "4_blocks":
                    four_blocks.append(block_dictionary[treatment][block_key][0])
                elif block_key == "3_speaker":
                    three_speaker_blocks.append(block_dictionary[treatment][block_key][0])                    
                else:
                    three_listener_blocks.append(block_dictionary[treatment][block_key][0])
                block_dictionary[treatment][block_key] = block_dictionary[treatment][block_key][1:]

        # Shuffle the blocks
        random.shuffle(four_blocks)
        random.shuffle(three_listener_blocks)
        random.shuffle(three_speaker_blocks)

        # Create the normal game without the attention check
        curr_blocks = []
        for i in range(len(treatment_order)):
            curr_four_block = four_blocks[i]
            curr_role = curr_four_block["id"][1]
            if curr_role == 0: # Listener start
                curr_blocks.append(four_blocks[i])
                curr_blocks.append(three_speaker_blocks[i])
                curr_blocks.append(three_listener_blocks[i])
            else:
                curr_blocks.append(four_blocks[i])
                curr_blocks.append(three_listener_blocks[i])
                curr_blocks.append(three_speaker_blocks[i])

        # Add the attention check in there
        formatted_attn_check = sample_formatted_attn_check(tangrams_to_idx)
        attn_idx = random.sample(list(range(len(curr_blocks))), 1)[0]
        curr_blocks = curr_blocks[:attn_idx] + [formatted_attn_check] + curr_blocks[attn_idx:]

        return_json = {"blocks" : curr_blocks}
        all_jsons.append(return_json)

    return all_jsons

def sample_formatted_attn_check(tangrams_to_idx):
    formatted_attention_check = {}

    attention_check = random.sample(COMPLEX_ALL_ATTN_CHECKS, 1)[0]
    formatted_attention_check["img"] = attention_check["img"]
    formatted_attention_check["tgt"] = attention_check["tgt"]

    block_size = 10
    tangram_range = list(range(block_size))
    tangrams_order = [random.sample(tangram_range, block_size),
                      random.sample(tangram_range, block_size)]

    formatted_attention_check['order'] = tangrams_order
    formatted_attention_check["roles"] = ROLES[1]
    formatted_attention_check["anno"] = attention_check["anno"]
    formatted_attention_check["bot_treatment"] = "full"

    return formatted_attention_check

def get_pragmatic_context(curr_corpus, tangrams_to_idx, clip_model, args):
    # Initialize the lists needed for generation
    overall_context = []
    base_tangrams = []
    individual_blocks = []

    # Initialize the parameters for generation
    block_sizes = evenly_spread_values(args.block_size, args.similarity_blocks_per_block)
    
    for i in range(args.similarity_blocks_per_block):
        # Sample the base tangram
        base_tangram = sample_similarity_block_base(curr_corpus, clip_model, overall_context, args)
        base_tangrams.append(base_tangram)

        # Sample the similarity block
        similarity_block = sample_similarity_block(curr_corpus, base_tangram, block_sizes[i], clip_model, args)
        individual_blocks.append(similarity_block)
        overall_context.extend(similarity_block)

        # Filter out the corpus
        curr_corpus = [tangram for tangram in curr_corpus if tangram not in overall_context]

    # Sample the targets at random
    targets = random.sample(overall_context, args.targets_per_block)

    # Construct the dictionary
    context_dict = {
        "img" : [tangrams_to_idx[t] for t in overall_context],
        "tgt" : [tangrams_to_idx[t] for t in targets],
        "similarity_blocks" : [
            [tangrams_to_idx[t] for t in curr_block] for curr_block in individual_blocks
        ]
    }

    return context_dict

def evenly_spread_values(block_size, num_similarity_blocks):
    sim_block_sizes = [0 for _ in range(num_similarity_blocks)]
    for i in range(block_size):
        idx = i % num_similarity_blocks
        sim_block_sizes[idx] += 1
    return sim_block_sizes

def sample_similarity_block_base(curr_corpus, clip_model, overall_context, args):
    # Get list of candidate tangrams
    candidate_base_tangrams = get_candidate_base_tangrams(curr_corpus, clip_model,
                                                          overall_context, args)
    if len(candidate_base_tangrams) == 0:
        print("Ran out of base tangram candidates!")
        assert(False)

    base_tangram = random.sample(candidate_base_tangrams, 1)[0]
    return base_tangram

def get_candidate_base_tangrams(curr_corpus, clip_model, overall_context, args):
    candidate_base_tangrams = []
    for tangram in curr_corpus:
        if valid_base_tangram(overall_context, tangram, clip_model, args):
            candidate_base_tangrams.append(tangram)
    return candidate_base_tangrams

def valid_base_tangram(overall_context, tangram, clip_model, args):
    for context_tangram in overall_context:
        if clip_model[context_tangram[:-4]][tangram[:-4]] > args.block_distance:
            return False
    return True

def sample_similarity_block(curr_corpus, base_tangram, similarity_block_size,
                            clip_model, args):
    # Get the most similar tangrams to the base tangram
    base_similarities = clip_model[base_tangram[:-4]]
    sorted_similarities = sorted(base_similarities.items(), reverse=True, key=lambda x: x[1])
    sorted_similarities = [sim for sim in sorted_similarities if sim[0] + ".svg" in curr_corpus]

    # Separate out the tangrams and the scores
    sorted_tangrams = [sim[0] + ".svg" for sim in sorted_similarities]
    sorted_scores = [sim[1] for sim in sorted_similarities]
    k = similarity_block_size - 1

    if args.sampling_method == "top_k":
        similarity_block = [base_tangram] + sorted_tangrams[:k]
    else:
        distribution = get_similarity_distribution(sorted_scores, args.sampling_temperature)
        sampled_indices = sample_without_replacement(distribution, k)
        similarity_block = [base_tangram] + [sorted_tangrams[i] for i in sampled_indices]

    if len(similarity_block) != similarity_block_size:
        print(similarity_block, similarity_block_size)
        assert(len(similarity_block) == similarity_block_size)

    return similarity_block

def get_similarity_distribution(scores, temperature):
    logits = torch.Tensor([score / temperature for score in scores])
    probs = F.softmax(logits, dim=0)
    return probs

def sample_without_replacement(distribution, K):
    new_distribution = torch.clone(distribution)

    samples = []
    for i in range(K):
        current_sample = torch.multinomial(new_distribution, num_samples=1).item()
        samples.append(current_sample)

        new_distribution[current_sample] = 0
        new_distribution = new_distribution / torch.sum(new_distribution)

    return samples

def generate_json(trial_list, json_folder, json_filename, comment, 
                  game_num, subfolder_name):
    # Initialize the folder for the json
    mkdir(json_folder)
    filepath = os.path.join(json_folder, json_filename)

    with open(filepath, 'w') as f:
        json.dump(trial_list, f)

def get_data(restricted_dataset=""):
    # Get the list of all paths
    if restricted_dataset == "":
        paths = os.listdir(EMPTY_DATA_PATH)
    else:
        with open(restricted_dataset, 'rb') as f:
            paths = pickle.load(f)
        paths = [path + ".svg" for path in paths]
    paths = [path for path in paths if ".DS_Store" not in path]
    random.shuffle(paths)

    # Remove duplicates
    for duplicate in ['page6-51.svg', 'page6-66.svg', 'page4-170.svg']:
        if duplicate in paths:
            paths.remove(duplicate)

    print(f"There are {len(paths)} tangrams total")
    return paths

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def main():
    # Get the arguments for game generation
    args = get_args()

    # Get the tangram to idx mapping for compression
    t2idx_path = os.path.join(JSON_PATH, 'tangram_to_idx.json')
    with open(t2idx_path, 'r') as f:
        tangrams_to_idx = json.load(f)

    # Load the CLIP model for similarity
    with open(CLIP_PATH, 'rb') as f:
        clip_model = pickle.load(f)

    # Generate games
    # Generate json for all games' selection states
    json_folder = os.path.join(JSON_PATH, args.subfolder_name)
    mkdir(json_folder)
    tangram_corpus = get_data(restricted_dataset=args.restricted_data_file)
    
    block_dictionary = get_all_complete_game_blocks(tangram_corpus, tangrams_to_idx, clip_model, args)
    regrouped_jsons = regroup_complete_game_blocks(block_dictionary, tangrams_to_idx, args)

    for i, regrouped_json in enumerate(regrouped_jsons):
        # Save the interaction as a json
        json_filename = f'game_json_{i}.json'
        generate_json(regrouped_json, json_folder, json_filename,
                      args.comment, i, args.subfolder_name) 

if __name__ == "__main__":
    main()
