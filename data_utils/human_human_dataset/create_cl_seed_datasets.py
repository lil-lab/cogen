import random
import numpy as np
import os
import argparse
import pickle
import json

REFGAME_FOLDER = "game_server_side/refgame"
SPLIT_DATA_FOLDER = "/home/mog29/cogen/data_and_checkpoints/continual_learning"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str)
    parser.add_argument("--statistic", type=str)
    parser.add_argument("--treatment", type=str)
    return parser.parse_args()

def get_all_games():
    filepath = "game_server_side/game_visualization/all_data_saved.pkl"
    with open(filepath, 'rb') as f:
        game_dict, player_dict = pickle.load(f)
    all_games = game_dict["oct_30"]
    return all_games
    
def get_similarity_block(block, target_img):
    sim_block_num = -1
    for i, sim_block in enumerate(block["similarity_blocks"]):
        if target_img in sim_block:
            sim_block_num = i
    assert(sim_block_num != -1)

    # Return the similarity block
    to_return = block["similarity_blocks"][sim_block_num]
    to_return = [img[:-4] for img in to_return]
    return to_return

def get_index_to_context(config_data):
    index_to_context = {}
    curr_index = 0
    for block in config_data["blocks"]:
        curr_context = tuple(block["id"])
        num_targets = len(block["tgt"])
        for i in range(num_targets):
            sim_block = get_similarity_block(block, block["tgt"][i])
            index_to_context[curr_index + i] = (curr_context, sim_block)
        curr_index += num_targets

    return index_to_context

def get_reformatted_json(config, idx_to_tangram):
    # Get the raw data
    with open(os.path.join(REFGAME_FOLDER, config), 'r') as f:
        game_json = json.load(f)
        
    # Reformat the images and targets
    updated_game_json = {"blocks" : []}
    for block in game_json["blocks"]:
        new_block = {
            "img" : [idx_to_tangram[str(img)] for img in block["img"]],
            "tgt" : [idx_to_tangram[str(img)] for img in block["tgt"]],
            "id" : block["id"],
            "similarity_blocks" : []
        }
    
        for sim_block in block["similarity_blocks"]:
            new_block["similarity_blocks"].append([idx_to_tangram[str(img)] for img in sim_block])

        updated_game_json["blocks"].append(new_block)

    return updated_game_json
    
def get_context_to_rounds(anno_games):
    i2t_path = os.path.join(REFGAME_FOLDER, 'public', 'games', 'idx_to_tangram.json')
    with open(i2t_path) as f:
        idx_to_tangram = json.load(f)

    context_to_stats = {}
    for game_id, game_dict in anno_games.items():
        if game_dict["status"] != "complete":
            continue
        if game_id == "JjZrDALqAg5qHp7tg":
            continue

        # Get the round_idx to context mapping 
        reformatted_json = get_reformatted_json(game_dict["config"], idx_to_tangram)
        index_to_context = get_index_to_context(reformatted_json)

        # Iterate over each round
        for curr_round in game_dict["roundDicts"]:
            curr_context, sim_block = index_to_context[curr_round["index"]]
            curr_round["similarity_block"] = sim_block
            if curr_context not in context_to_stats:
                context_to_stats[curr_context] = {"total" : 0, "total_complete": 0, "correct" : 0, "rounds" : []}

            context_to_stats[curr_context]["total"] += 1
            if curr_round["selection"] != "no_clicks":
                context_to_stats[curr_context]["total_complete"] += 1
                if curr_round["selection"] == curr_round["target"]:
                    context_to_stats[curr_context]["correct"] += 1
            context_to_stats[curr_context]["rounds"].append(curr_round)

    return context_to_stats

def skip_round(curr_round):
    # Is it idle or incorrect?
    return curr_round["selection"] != curr_round["target"]

def get_distractor_annos(rounds, round_index):
    distractor_annos = []
    for curr_index, curr_round in enumerate(rounds):
        if curr_index == round_index:
            continue
        distractor_annos.append(curr_round["chat"])
    return distractor_annos

def save_games(context_to_rounds, contexts, split):
    # Create a similar game_id to rounds structure
    save_dict = {}
    for context in contexts:
        game_id, _ = context
        game_dict = {}
        rounds = context_to_rounds[context]["rounds"]

        for round_index, curr_round in enumerate(rounds):
            if skip_round(curr_round):
                continue
            distractor_annos = get_distractor_annos(rounds, round_index)

            round_dict = {
                "speaker_context" : [targ[:-4] for targ in curr_round["speaker_context"]],
                "listener_context" : [targ[:-4] for targ in curr_round["listener_context"]],
                "chat" : curr_round["chat"],
                "gt_target" : curr_round["target"][:-4],
                "human_target" : curr_round["selection"][:-4],
                "distractor_annos" : distractor_annos,
                "similarity_block" : curr_round["similarity_block"],
                "speaker" : curr_round["speaker"],
                "listener" : curr_round["listener"]
            }
            game_dict[round_index] = round_dict

        if len(game_dict) > 0:
            save_dict[game_id] = game_dict

    # Save the dict
    if split == "train":
        savepath = os.path.join(SPLIT_DATA_FOLDER, f"cl_r0.json")
    else:
        savepath = os.path.join(SPLIT_DATA_FOLDER, f"cl_val.json")        
    with open(savepath, 'w') as f:
        json.dump(save_dict, f)

def main():
    # Get the games
    all_games = get_all_games()
    context_to_rounds = get_context_to_rounds(all_games)

    # Perform the splitting
    # Train contexts: Sample 25 from complete games
    complete_contexts = [ctx for ctx, ctx_info in context_to_rounds.items() if ctx_info["total_complete"] == 10]
    tr_contexts = random.sample(complete_contexts, 25)
    save_games(context_to_rounds, tr_contexts, "train")

    # Val contexts: The rest
    val_contexts = [ctx for ctx in context_to_rounds if ctx not in tr_contexts]
    save_games(context_to_rounds, val_contexts, "val")

    # Print number of annotations
    for contexts, split in zip([tr_contexts, val_contexts], ["train", "val"]):
        total = 0
        total_correct = 0
        for context in contexts:
            total += context_to_rounds[context]["total_complete"]
            total_correct += context_to_rounds[context]["correct"]

        print(f"{split} split: {total} annotations, {total_correct} correct")

if __name__ == "__main__":
    main()
