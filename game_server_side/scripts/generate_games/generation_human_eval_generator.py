import os
import json
import pickle
import random
import argparse
from tqdm import tqdm

from config_generator import sample_similarity_block_base, sample_similarity_block, mkdir, evenly_spread_values


ROOT_PATH = "/Users/mustafaomergul/Desktop/Cornell/Research/tangrams-compgen/tangrams-ref-dev-omer"
DIVERGENCE_PATH = ROOT_PATH + "/scripts/generate_games/full_SND.json"
EMPTY_DATA_PATH =  "/Users/mustafaomergul/Desktop/Cornell/Research/kilogram/dataset/tangrams-svg"
JSON_PATH = ROOT_PATH + "/refgame/public/games"
HUMAN_EVAL_PATH = ROOT_PATH + "/debugging_human_eval_data"
CLIP_PATH = "/Users/mustafaomergul/Desktop/Cornell/Research/tangrams-compgen/similarity_files/clip_model0_similarities.pkl"
ROLES = [["speaker", "listener"], ["listener", "speaker"]]


def get_args():
    parser = argparse.ArgumentParser(description="Generating jsons for tangram game trials")

    # Interaction arguments
    parser.add_argument('--eval_type', type=str, choices=["isolated", "similar"],
                        help="What type of context to construct")
    parser.add_argument('--maximum_snd_percentile', type=float, default=-1,
                        help="Sample targets from the bottom half of SND")
    parser.add_argument('--minimum_snd_percentile', type=float, default=-1,
                        help="Sample targets from the top half of SND")
    parser.add_argument('--num_contexts', type=int, default=50,
                        help="Number of contexts to generate")
    parser.add_argument('--max_similarity', type=float, default=0,
                        help="Maximum similarity distractors should have to the target")

    parser.add_argument('--sampling_method', type=str, default="sample",
                        choices=["top_k", "sample"], help="How to sample similar tangrams")
    parser.add_argument('--sampling_temperature', type=float, default=0.055,
                        help="The softmax temperature for when we sample the similarity block")
    parser.add_argument('--block_distance', type=float, default=1,
                        help="How distant should the base tangram of past blocks be to the current base?")

    parser.add_argument('--game_id_prefix', type=str,
                        help="How will we disambiguate between contexts across multiple runs?")

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

def get_relevant_snd_data(max_percentile, min_percentile):
    assert(not (max_percentile == -1 and min_percentile == -1))

    # Get sorted tangram-snd pairings
    with open(DIVERGENCE_PATH, 'r') as f:
        snd_map = json.load(f)
    snd_list = [(tangram, snd) for tangram, snd in snd_map.items()]
    snd_list = sorted(snd_list, key=lambda x: x[1])

    if max_percentile != -1:
        slice_len = int(len(snd_list) * max_percentile)
        sliced_list = snd_list[:slice_len]
    else:
        slice_len = int(len(snd_list) * min_percentile)
        sliced_list = snd_list[slice_len:]

    return_tangrams = [tangram for tangram, _ in sliced_list]
    return return_tangrams

def construct_isolated_context(args, all_data, target_candidates, tangrams_to_idx, similarity_scores):
    context_dict = {}

    # Sample attention check target
    target = random.sample(target_candidates, 1)[0] + ".svg"
    targets = [target]
    overall_context = [target]
    
    # Filter out data to exclude those too similar to the attention check
    filtered_data = get_filtered_tangrams(similarity_scores, all_data, target, args.max_similarity)

    # Get the similarity blocks
    for i in [3, 3]:
        # Sample base tangram
        base_tangram = sample_similarity_block_base(filtered_data, similarity_scores, overall_context, args)

        # Sample the similarity block
        similarity_block = sample_similarity_block(filtered_data, base_tangram, i, similarity_scores, args)
        overall_context.extend(similarity_block)

        # Filter out the corpus
        filtered_data = [tangram for tangram in filtered_data if tangram not in overall_context]

    # Get the random block
    overall_context.extend(random.sample(filtered_data, 3))

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

def construct_difficult_context(args, all_data, target_candidates, tangrams_to_idx, similarity_scores):
    context_dict = {}

    # Sample attention check target
    target = random.sample(target_candidates, 1)[0] + ".svg"
    overall_context = []
    individual_blocks = []
    
    # Iterate over each similarity block
    block_sizes = evenly_spread_values(10, 3)
    for i in range(3):
        # Sample the base tangram
        base_tangram = target if i == 0 else sample_similarity_block_base(all_data, similarity_scores, overall_context, args)

        # Sample the similarity block
        similarity_block = sample_similarity_block(all_data, base_tangram, block_sizes[i], similarity_scores, args)
        individual_blocks.append(similarity_block)
        overall_context.extend(similarity_block)
        
        # Filter out the corpus
        all_data = [tangram for tangram in all_data if tangram not in overall_context]

    # Make the targets the first similarity block
    targets = individual_blocks[0]

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

def get_filtered_tangrams(similarity_scores, all_data, target, max_similarity):
    filtered_tangrams = []
    tangram_scores = similarity_scores[target[:-4]]

    for tangram, score in tangram_scores.items():
        svg_tangram = tangram + ".svg"
        if svg_tangram in all_data and score < max_similarity:
            filtered_tangrams.append(svg_tangram)

    return filtered_tangrams

def save_contexts(contexts, args):
    to_save = {
        "name" : args.game_id_prefix,
        "contexts" : contexts
    }

    mkdir(HUMAN_EVAL_PATH)
    path = os.path.join(HUMAN_EVAL_PATH, f"{args.game_id_prefix}.json")
    with open(path, 'w') as f:
        json.dump(to_save, f)

if __name__ == "__main__":
    # Get the arguments for attention check generation
    args = get_args()

    # Load the CLIP model for similarity
    t2idx_path = os.path.join(JSON_PATH, 'tangram_to_idx.json')
    with open(t2idx_path, 'r') as f:
        tangrams_to_idx = json.load(f)
    with open(CLIP_PATH, 'rb') as f:
        similarity_scores = pickle.load(f)
    all_data = get_data()
    target_candidates = get_relevant_snd_data(args.maximum_snd_percentile, args.minimum_snd_percentile)

    # Generate the games
    contexts = []
    construction_func = construct_isolated_context if args.eval_type == "isolated" else construct_difficult_context
    for i in range(args.num_contexts):
        contexts.append(construction_func(args, all_data, target_candidates, tangrams_to_idx, similarity_scores))
    save_contexts(contexts, args)

