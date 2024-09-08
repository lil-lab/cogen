# File: create_cl_round_datasets.py
# ---------------------------------
# Main preprocessing script for converting the initial data processed from MongoDB into
# json files for each model. Should be called after deployment.

import random
import numpy as np
import os
import argparse
import pickle
import json

REFGAME_FOLDER = "game_server_side/refgame"
SPLIT_DATA_FOLDER = "/home/mog29/cogen/data_and_checkpoints/continual_learning"
TREATMENTS = ["full", "baseline", "no_ji", "no_ds"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str)
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    assert(args.annotation is not None and args.round is not None)
    return args

def treatment_to_data_sharing(treatment):
    return treatment not in ["baseline", "no_ds"]

def get_all_games():
    filepath = "game_server_side/game_visualization/all_data_saved.pkl"
    with open(filepath, 'rb') as f:
        game_dict, _ = pickle.load(f)
    return game_dict
    
def get_annotation_games(all_games, annotation):
    anno_games = all_games[annotation]
    new_games = {}

    for game_id, game_dict in anno_games.items():
        if game_dict["status"] != "complete":
            continue
        new_games[game_id] = game_dict

    return new_games

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

def get_index_to_context(config_data, round_idx):
    index_to_context = {}
    curr_index = 0
    for block in config_data["blocks"]:
        if "id" not in block:
            curr_index += 1
            continue

        game_id, role_idx = block["id"]
        treatment = block["bot_treatment"]
        role_name = "listener" if role_idx == 0 else "speaker"
        context_name = f"r{round_idx}_{treatment}_{role_name}_{game_id}"

        num_targets = len(block["tgt"])
        for i in range(num_targets):
            sim_block = get_similarity_block(block, block["tgt"][i])
            index_to_context[curr_index + i] = (context_name, sim_block)
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
            "similarity_blocks" : []
        }

        if "id" in block:
            new_block["id"] = block["id"]
            new_block["bot_treatment"] = block["bot_treatment"]
            for sim_block in block["similarity_blocks"]:
                new_block["similarity_blocks"].append([idx_to_tangram[str(img)] for img in sim_block])

        updated_game_json["blocks"].append(new_block)

    return updated_game_json
    
def skip_round(curr_round, bot_role):
    if curr_round["isAttnCheck"]:
        return True

    if bot_role == "listener":
        # Check whether the speaker idled
        return curr_round["chat"] == "Speaker idled"
    else:
        # Check whether the speaker idled or did not send anything
        message = curr_round["chat"]
        selection = curr_round["selection"]
        if message in ["Speaker failed?", "Speaker idled"]:
            return True
        elif selection == "no_clicks":
            return True
        elif "reportedGameId" in curr_round:
            game_match = curr_round["reportedGameId"] == curr_round["game_id"]
            round_match = curr_round["reportedRoundId"] == curr_round["round_id"]
            return not (game_match and round_match)
        else:
            return False

def skip_human_round(curr_round):
    if curr_round["isAttnCheck"]:
        return True
    return curr_round["selection"] != curr_round["target"]

def get_distractor_annos(rounds, round_index):
    distractor_annos = []
    for curr_index, curr_round in enumerate(rounds):
        if curr_index == round_index:
            continue
        distractor_annos.append(curr_round["chat"])
    return distractor_annos

## MAIN FUNCTIONS ##

def get_context_to_rounds(annotation_games, round_idx):
    i2t_path = os.path.join(REFGAME_FOLDER, 'public', 'games', 'idx_to_tangram.json')
    with open(i2t_path) as f:
        idx_to_tangram = json.load(f)

    context_to_rounds = {}
    for game_id, game_dict in annotation_games.items():
        # Get the round_idx to context mapping 
        reformatted_json = get_reformatted_json(game_dict["config"], idx_to_tangram)
        index_to_context = get_index_to_context(reformatted_json, round_idx)

        # Iterate over each round
        for curr_round in game_dict["roundDicts"]:
            curr_treatment = curr_round["bot_treatment"]
            if curr_treatment == "human":
                if skip_human_round(curr_round):
                    continue
            else:
                bot_role = "listener" if curr_round["listener"] in TREATMENTS else "speaker"
                if skip_round(curr_round, bot_role):
                    continue
            context_name, sim_block = index_to_context[curr_round["index"]]
            curr_round["similarity_block"] = sim_block

            if curr_treatment not in context_to_rounds:
                context_to_rounds[curr_treatment] = {}
            if context_name not in context_to_rounds[curr_treatment]:
                context_to_rounds[curr_treatment][context_name] = []
            context_to_rounds[curr_treatment][context_name].append(curr_round)

    return context_to_rounds

def extract_treatment_c2r(context_to_rounds, treatment, round_idx):
    if treatment in ["baseline", "no_ds"] and round_idx == 1:
        round_treatment = "full" if treatment == "no_ds" else "no_ji"
        c2r = {}
        for context_name, context_rounds in context_to_rounds[round_treatment].items():
            bot_role = "speaker" if "speaker" in context_name else "listener"
            game_id = context_name.split("_")[-1]
            new_context_name = f'r{round_idx}_{treatment}_{bot_role}_{game_id}'
            c2r[new_context_name] = context_rounds
        return c2r
    else:
        return context_to_rounds[treatment]

def save_games(args, context_to_rounds, contexts, treatment, round_idx, suffix):
    # Create a similar game_id to rounds structure
    save_dict = {}

    for context_name in contexts:
        rounds = context_to_rounds[context_name]
        game_dict = {}
        bot_role = "listener" if "listener" in context_name else "speaker"

        for round_index, curr_round in enumerate(rounds):
            distractor_annos = get_distractor_annos(rounds, round_index)
            reward = 1 if curr_round["selection"] == curr_round["target"] else -1

            # Edge case to handle error in data collection
            if bot_role == "listener" and treatment != "human":
                if curr_round["selection"] not in curr_round["listener_context"]:
                    selection = "redo"
                elif "reportedGameId" in curr_round:
                    game_match = curr_round["reportedGameId"] == curr_round["game_id"]
                    round_match = curr_round["reportedRoundId"] == curr_round["round_id"]
                    if game_match and round_match:
                        selection = curr_round["selection"][:-4]
                    else:
                        selection = "redo"
                else:
                    selection = curr_round["selection"][:-4]
            else:
                selection = curr_round["selection"][:-4]

            round_dict = {
                "speaker_context" : [targ[:-4] for targ in curr_round["speaker_context"]],
                "listener_context" : [targ[:-4] for targ in curr_round["listener_context"]],
                "chat" : curr_round["chat"],
                "gt_target" : curr_round["target"][:-4],
                "selection" : selection, 
                "reward" : reward,
                "round" : round_idx,
                "distractor_annos" : distractor_annos,
                "similarity_block" : curr_round["similarity_block"],
                "speaker" : curr_round["speaker"],
                "listener" : curr_round["listener"]
            }
            game_dict[round_index] = round_dict

        if len(game_dict) > 0:
            save_dict[context_name] = game_dict

    # Save the dict
    if treatment == "human":
        savepath = os.path.join(SPLIT_DATA_FOLDER, f"cl_r{round_idx}_{treatment}_{suffix}.json")
    else:
        savepath = os.path.join(SPLIT_DATA_FOLDER, f"cl_r{round_idx}_{treatment}_{suffix}_unprocessed.json")        
    with open(savepath, 'w') as f:
        json.dump(save_dict, f)

def main():
    args = get_args()
    all_games = get_all_games()

    # First get the mapping from contexts to rounds for each treatment
    annotation_games = get_annotation_games(all_games, args.annotation)
    context_to_rounds = get_context_to_rounds(annotation_games, args.round)

    for treatment in TREATMENTS:
        c2r = extract_treatment_c2r(context_to_rounds, treatment, args.round)

        share_data = treatment_to_data_sharing(treatment)
        if share_data:
            save_context = c2r.keys()
            save_games(args, c2r, save_context, treatment, args.round, "all")
        else:
            listener_contexts = [ctx for ctx in c2r if "listener" in ctx]
            save_games(args, c2r, listener_contexts, treatment, args.round, "listener")            
            speaker_contexts = [ctx for ctx in c2r if "speaker" in ctx]
            save_games(args, c2r, speaker_contexts, treatment, args.round, "speaker")

    # Create the human games
    treatment = "human"
    c2r = extract_treatment_c2r(context_to_rounds, treatment, args.round) 
    save_context = c2r.keys()
    save_games(args, c2r, save_context, treatment, args.round, "all")
    

if __name__ == "__main__":
    main()
