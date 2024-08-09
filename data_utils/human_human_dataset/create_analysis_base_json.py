import random
import numpy as np
import os
import argparse
import pickle
import json

REFGAME_FOLDER = "/home/mog29/cogen/game_server_side/refgame"
SPLIT_DATA_FOLDER = "/home/mog29/cogen/data_and_checkpoints/continual_learning"
TREATMENTS = ["full", "baseline", "no_ji", "no_ds"]
ANNOTATIONS = ["may_1", "may_7", "may_13", "may_17"]

def get_all_games():
    filepath = "game_server_side/game_visualization/all_data_saved.pkl"
    with open(filepath, 'rb') as f:
        game_dict, _ = pickle.load(f)
    return game_dict
    
def get_human_games(all_games):
    new_games = {}

    for annotation in ANNOTATIONS:
        for game_id, game_dict in all_games[annotation].items():
            if game_dict["status"] != "complete":
                continue
            if game_dict["treatment"] != "human_human":
                continue
            new_games[game_id] = game_dict

    return new_games

def get_annotation_games(all_games, annotation):
    anno_games = all_games[annotation]
    new_games = {}

    for game_id, game_dict in anno_games.items():
        if skip_game_for_annotation(game_dict, annotation):
            continue
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
    return curr_round["selection"] not in curr_round["listener_context"]


## MAIN FUNCTIONS ##

def get_context_to_rounds(human_games):
    i2t_path = os.path.join(REFGAME_FOLDER, 'public', 'games', 'idx_to_tangram.json')
    with open(i2t_path) as f:
        idx_to_tangram = json.load(f)

    context_to_rounds = {}
    all_rounds = set()
    for game_id, game_dict in human_games.items():
        # Get the round_idx to context mapping 
        round_idx = ANNOTATIONS.index(game_dict['anno']) + 1
        all_rounds.add(round_idx)
        
        reformatted_json = get_reformatted_json(game_dict["config"], idx_to_tangram)
        index_to_context = get_index_to_context(reformatted_json, round_idx)

        # Iterate over each round
        for curr_round in game_dict["roundDicts"]:
            if skip_human_round(curr_round):
                continue
            context_name, sim_block = index_to_context[curr_round["index"]]
            curr_round["similarity_block"] = sim_block

            if context_name not in context_to_rounds:
                context_to_rounds[context_name] = []
            context_to_rounds[context_name].append(curr_round)

    return context_to_rounds

def get_context_round(context_name):
    context_round_string = context_name.split("_")[0]
    for i in range(1, 5):
        if f"r{i}" == context_round_string:
            return i
    assert(False)

def save_games(context_to_rounds):
    # Create a similar game_id to rounds structure
    save_dict = {}

    num_added = 0
    for context_name in context_to_rounds:
        rounds = context_to_rounds[context_name]
        game_dict = {}

        for round_index, curr_round in enumerate(rounds):
            reward = 1 if curr_round["selection"] == curr_round["target"] else -1
            selection = curr_round["selection"][:-4]
            round_dict = {
                "speaker_context" : [targ[:-4] for targ in curr_round["speaker_context"]],
                "listener_context" : [targ[:-4] for targ in curr_round["listener_context"]],
                "chat" : curr_round["chat"],
                "gt_target" : curr_round["target"][:-4],
                "selection" : selection, 
                "reward" : reward,
                "round" : get_context_round(context_name),
                "similarity_block" : curr_round["similarity_block"],
                "speaker" : curr_round["speaker"],
                "listener" : curr_round["listener"]
            }
            game_dict[round_index] = round_dict

        if len(game_dict) > 0:
            save_dict[context_name] = game_dict
            num_added += len(game_dict)

    # Save the dict
    savepath = os.path.join(SPLIT_DATA_FOLDER, "analysis", "human_human.json")
    with open(savepath, 'w') as f:
        json.dump(save_dict, f)

    print(f"We have {num_added} utterances")

def main():
    all_games = get_all_games()
    human_games = get_human_games(all_games) 
    context_to_rounds = get_context_to_rounds(human_games)
    save_games(context_to_rounds)

if __name__ == "__main__":
    main()
