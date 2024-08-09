import numpy as np
from pymongo import MongoClient
import os
import argparse
import pickle
import json
from nltk.tokenize import word_tokenize
from copy import deepcopy

import spacy
nlp = spacy.load("en_core_web_sm")

REFGAME_FOLDER = "../../refgame"
TREATMENTS = ["human", "full", "baseline", "no_ji", "no_ds", "old_full"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str)
    parser.add_argument("--statistic", type=str)
    parser.add_argument("--treatment", type=str)
    return parser.parse_args()

def get_all_cached():
    filepath = "../../game_visualization/all_data_saved.pkl"
    with open(filepath, 'rb') as f:
        game_dict, player_dict = pickle.load(f)
    return game_dict, player_dict
    
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

def get_index_to_context(config_data):
    index_to_context = {}

    curr_index = 0
    for block in config_data["blocks"]:
        if "id" not in block:
            continue
        curr_context = tuple(block["id"])
        treatment = block["bot_treatment"]
        num_targets = len(block["tgt"])
        for i in range(num_targets):
            index_to_context[curr_index + i] = (curr_context, treatment)
        curr_index += num_targets

    return index_to_context

def get_treatment_games(all_games, annotation, treatment):
    anno_games = all_games[annotation]

    # Get filtered set of games with filtered sets of rounds
    new_games = {}
    for game_id, game_dict in anno_games.items():
        if game_dict["status"] != "complete":
            continue

        # Construct filtered game
        new_game_dict = deepcopy(game_dict)
        new_round_dicts = []
        for curr_round in game_dict['roundDicts']:
            if "anno" in curr_round:
                continue

            curr_treatment = curr_round['bot_treatment']
            if curr_treatment != treatment:
                continue

            new_round_dicts.append(curr_round)

        # Overwrite the round dicts
        new_game_dict['roundDicts'] = new_round_dicts
        if len(new_round_dicts) == 0:
            continue

        # Overwrite the players
        for i in range(2):
            if new_game_dict['players'][i] == 'human_model':
                new_game_dict['players'][i] = treatment

        new_games[game_id] = new_game_dict
    return new_games

def get_list_of_players(games):
    all_players = set()
    for game_id, game_dict in games.items():
        for player in game_dict["players"]:
            all_players.add(player)
    return all_players

def player_messages(games, player):
    messages = []
    for _, game_dict in games.items():
        if player not in game_dict["players"]:
            continue

        for curr_round in game_dict["roundDicts"]:
            if curr_round["speaker"] != player:
                continue
            if curr_round["isAttnCheck"]:
                continue
            if curr_round["msgLen"] == -1:
                continue
            messages.append(curr_round["chat"])
    return messages

def tokenize_message(message):
    message = message.lower().strip()
    tokens = word_tokenize(message)
    return tokens

def get_vocabulary(messages):
    vocabulary = set()
    for message in messages:
        message_tokens = tokenize_message(message)
        vocabulary = vocabulary.union(set(message_tokens))
    return vocabulary

def get_average_message_length(messages):
    message_lens = []
    for message in messages:
        tokens = tokenize_message(message)
        message_lens.append(len(tokens))
    return np.mean(message_lens)

def get_dependency_metric_for_message(message):
    message = message.lower().strip()
    doc = nlp(message)

    # First get out degrees
    out_degrees = []
    root = None
    for token in doc:
        if len([ancestor for ancestor in token.ancestors]) == 0:
            root = token

        children = [child for child in token.children]
        if len(children) == 0:
            continue

        out_degrees.append(len(children))

    max_width = max(out_degrees) if len(out_degrees) != 0 else 0
    average_branching = np.mean(out_degrees) if len(out_degrees) != 0 else 0
    
    # Next get max depth
    max_depth = get_height(root)
    
    return max_depth, max_width, average_branching

def get_height(node):
    if len([child for child in node.children]) == 0:
        return 0
    else:
        child_heights = [get_height(child) for child in node.children]
        return 1 + max(child_heights)    

def get_dependency_metrics(messages):
    max_depths = []
    max_widths = []
    branching_factors = []

    for message in messages:
        max_depth, max_width, branching_factor = get_dependency_metric_for_message(message)
        max_depths.append(max_depth)
        max_widths.append(max_width)
        branching_factors.append(branching_factor)

    return np.mean(max_depths), np.mean(max_widths), np.mean(branching_factors)    

def get_human_vocab_size(player_to_messages, treatment):
    full_vocab = set()
    for player, messages in player_to_messages.items():
        if player == treatment:
            continue
        curr_vocab = get_vocabulary(messages)
        full_vocab = full_vocab.union(curr_vocab)
    return len(full_vocab)

def get_bot_vocab_size(player_to_messages, treatment):
    bot_messages = player_to_messages[treatment]
    bot_vocab = get_vocabulary(bot_messages)
    return len(bot_vocab)

def get_human_message_length(player_to_messages, treatment):
    average_lens = []
    for player, messages in player_to_messages.items():
        if player == treatment:
            continue
        player_avg = get_average_message_length(messages)
        average_lens.append(player_avg)

    return np.mean(average_lens)

def get_bot_message_length(player_to_messages, treatment):
    bot_messages = player_to_messages[treatment]
    avg_bot_len = get_average_message_length(bot_messages)
    return avg_bot_len

def get_human_dependency_metrics(player_to_messages, treatment):
    max_depths = []
    max_widths = []
    branching_factors = []

    for player, messages in player_to_messages.items():
        if player == treatment:
            continue
        max_depth, max_width, branching_factor = get_dependency_metrics(messages)
        max_depths.append(max_depth)
        max_widths.append(max_width)
        branching_factors.append(branching_factor)

    return np.mean(max_depths), np.mean(max_widths), np.mean(branching_factors)

def get_bot_dependency_metrics(player_to_messages, treatment):
    bot_messages = player_to_messages[treatment]
    return get_dependency_metrics(bot_messages)

def player_accuracies(games, player, idx_to_tangram, role):
    num_listener_rounds = 0
    num_correct = 0

    for _, game_dict in games.items():
        if player not in game_dict["players"]:
            continue
    
        for i, curr_round in enumerate(game_dict["roundDicts"]):
            if curr_round[role] != player:
                continue
            if curr_round["isAttnCheck"]:
                continue
            if curr_round["selection"] not in curr_round["listener_context"]:
                continue
            if "reportedGameId" in curr_round:
                game_match = curr_round["reportedGameId"] == curr_round["game_id"]
                round_match = curr_round["reportedRoundId"] == curr_round["round_id"]
                if not (game_match and round_match):
                    print("Uh oh buddddy")
                    continue

            num_listener_rounds += 1
            num_correct += 1 if curr_round["selection"] == curr_round["target"] else 0

    return num_correct / num_listener_rounds

def get_human_accuracy(player_to_accuracies, treatment):
    accs = []
    for player, acc in player_to_accuracies.items():
        if player == treatment:
            continue
        accs.append(acc)
    return np.mean(accs)

def get_bot_accuracy(player_to_accuracies, treatment):
    return player_to_accuracies[treatment]

def get_overall_accuracy(games, player, role, avoid=True):
    num_listener_rounds = 0
    num_correct = 0

    for _, game_dict in games.items():
        for i, curr_round in enumerate(game_dict["roundDicts"]):
            if avoid and curr_round[role] == player:
                continue
            if not avoid and curr_round[role] != player:
                continue
            if curr_round["isAttnCheck"]:
                continue
            if curr_round["selection"] not in curr_round["listener_context"]:
                continue
            if "reportedGameId" in curr_round:
                game_match = curr_round["reportedGameId"] == curr_round["game_id"]
                round_match = curr_round["reportedRoundId"] == curr_round["round_id"]
                if not (game_match and round_match):
                    continue

            num_listener_rounds += 1
            num_correct += 1 if curr_round["selection"] == curr_round["target"] else 0

    return num_correct / num_listener_rounds
    
def player_hourly(games, player):
    pays = []

    for _, game_dict in games.items():
        if player not in game_dict["players"]:
            continue

        pays.append(game_dict["hourlyPay"])

    return np.mean(pays)

## MAIN FUNCTIONS ##

def report_data_collected(all_games, annotation):
    # First iterate over each game
    context_to_stats = {}
    for game_id, game_dict in all_games[annotation].items():
        if game_dict["status"] != "complete":
            continue
        
        # Get the round_idx to context mapping
        config = game_dict["config"]
        with open(os.path.join(REFGAME_FOLDER, config), 'r') as f:
            config_data = json.load(f)
        index_to_context = get_index_to_context(config_data)
        
        # Iterate over each round
        for curr_round in game_dict["roundDicts"]:
            curr_index = curr_round["index"]
            if curr_index not in index_to_context:
                continue

            curr_context, treatment = index_to_context[curr_index]
            if treatment not in context_to_stats:
                context_to_stats[treatment] = {}
            if curr_context not in context_to_stats[treatment]:
                context_to_stats[treatment][curr_context] = {"total" : 0, "total_complete": 0}
            context_to_stats[treatment][curr_context]["total"] += 1
            if curr_round["selection"] != "no_clicks":
                context_to_stats[treatment][curr_context]["total_complete"] += 1

    # Then iterate over each treatment
    for treatment, treatment_stats in context_to_stats.items():
        context_count = len(treatment_stats)
        complete_contexts = 0
        total_utterances = 0
        annotated_per = []
        for context, stats in treatment_stats.items():
            if stats["total"] == 10:
                complete_contexts += 1
            annotated_per.append(stats["total_complete"])
            total_utterances += stats["total_complete"]
        
        if len(annotated_per) == 0:
            continue

        print(f"Treatment {treatment}:")
        print(f"We have collected games for {context_count} contexts.")
        print(f"We have complete annotations for {complete_contexts} of these. The number of usable annotations is: {total_utterances}")        
        print(f"On average, we have {np.mean(annotated_per)} annotated examples for each. Worst case: {min(annotated_per)}")
        print(annotated_per)
        print()

def report_premature_games(all_games, annotation):
    # Report which games we missed due to a player idling
    treatment_to_contexts = {}
    for game_id, game_dict in all_games[annotation].items():
        if game_dict["status"] == "complete":
            continue

        treatment = game_dict['treatment']
        if treatment not in treatment_to_contexts:
            treatment_to_contexts[treatment] = []
        treatment_to_contexts[treatment].append(game_dict["config"])

    for treatment, missed_contexts in treatment_to_contexts.items():
        print(f"Missed games for treatment {treatment}")
        if len(missed_contexts) == 0:
            print("No missed games!")
        else:
            for context in missed_contexts:
                print(context)
        print()
        
def report_bot_language_metrics(games, treatment, players):
    print("Reporting language statistics for bot games")

    # First get mapping from player to their messages
    player_to_messages = {
        player : player_messages(games, player) for player in players
    }
    
    # Report vocabulary sizes
    human_vocabulary_size = get_human_vocab_size(player_to_messages, treatment)
    print(f"Human vocabulary size: {human_vocabulary_size}")
    bot_vocabulary_size = get_bot_vocab_size(player_to_messages, treatment)
    print(f"Bot vocabulary size: {bot_vocabulary_size}")

    # Report message length stats
    avg_human_message_length = get_human_message_length(player_to_messages, treatment)
    print(f"Average human message length: {avg_human_message_length}")
    avg_bot_message_length = get_bot_message_length(player_to_messages, treatment)
    print(f"Average bot message length: {avg_bot_message_length}")

    # Report sentence complexity stats
    human_avg_depth, human_avg_width, human_branching_factor = get_human_dependency_metrics(player_to_messages, treatment)
    print(f"Average human dependency metrics: max depth: {human_avg_depth}, max width: {human_avg_width}, branching factor: {human_branching_factor}")
    bot_avg_depth, bot_avg_width, bot_branching_factor = get_bot_dependency_metrics(player_to_messages, treatment)
    print(f"Average bot dependency metrics: max depth: {bot_avg_depth}, max width: {bot_avg_width}, branching factor: {bot_branching_factor}")

    saved_metrics = {
        'vocabulary_size' : bot_vocabulary_size,
        'message_length' : avg_bot_message_length,
        'max_depth' : bot_avg_depth,
        'max_width' : bot_avg_width,
        'branching_factor' : bot_branching_factor,
        'human_vocabulary_size' : human_vocabulary_size,
        'human_message_length' : avg_human_message_length,
        'human_max_depth' : human_avg_depth,
        'human_max_width' : human_avg_width,
        'human_branching_factor' : human_branching_factor,
    }
    return saved_metrics


def report_hh_language_metrics(games, players):
    print("Reporting language statistics for human-human games")

    # First get mapping from player to their messages
    player_to_messages = {
        player : player_messages(games, player) for player in players
    }

    # Report vocabulary sizes, message lengths and dependency parse results
    human_vocabulary_size = get_human_vocab_size(player_to_messages, "human")
    print(f"Human vocabulary size: {human_vocabulary_size}")
    avg_human_message_length = get_human_message_length(player_to_messages, "human")    
    print(f"Average human message length: {avg_human_message_length}")
    human_avg_depth, human_avg_width, human_branching_factor = get_human_dependency_metrics(player_to_messages, "human")
    print(f"Average human dependency metrics: max depth: {human_avg_depth}, max width: {human_avg_width}, branching factor: {human_branching_factor}")

    saved_metrics = {
        'vocabulary_size' : human_vocabulary_size,
        'message_length' : avg_human_message_length,
        'max_depth' : human_avg_depth,
        'max_width' : human_avg_width,
        'branching_factor' : human_branching_factor
    }
    return saved_metrics

def report_hh_accuracy(treatment_games, players, idx_to_tangram):
    # Report the various accuracy types of each player
    print("Reporting accuracy information for human-human games")

    player_to_accuracies = {
        player : player_accuracies(treatment_games, player, idx_to_tangram, "listener") for player in players
    }

    acc = get_human_accuracy(player_to_accuracies, "human")
    overall_acc = get_overall_accuracy(treatment_games, "human", "listener", avoid=True)
    print(f"Average accuracy (per_player): {acc * 100:.2f}%, overall: {overall_acc * 100:.2f}")

    saved_metrics = {
        'generation_accuracy' : overall_acc,
        'comprehension_accuracy' : overall_acc,
    }
    return saved_metrics

def report_bot_accuracy(treatment_games, treatment, players, idx_to_tangram):
    print("Reporting accuracy information for bot games")
    
    player_to_listener_accuracies = {
        player : player_accuracies(treatment_games, player, idx_to_tangram, "listener") for player in players
    }
    speaker_acc = get_human_accuracy(player_to_listener_accuracies, treatment)
    overall_speaker_acc = get_overall_accuracy(treatment_games, treatment, "listener", avoid=True)
    print(f"Average speaker accuracy (per player): {speaker_acc * 100:.2f}%, overall: {overall_speaker_acc * 100:.2f}")

    player_to_speaker_accuracies = {
        player : player_accuracies(treatment_games, player, idx_to_tangram, "speaker") for player in players
    }
    listener_acc = get_human_accuracy(player_to_speaker_accuracies, treatment)
    overall_listener_acc = get_overall_accuracy(treatment_games, treatment, "speaker", avoid=True)
    print(f"Average listener accuracy (per player): {listener_acc * 100:.2f}%, overall: {overall_listener_acc * 100:.2f}")

    saved_metrics = {
        'generation_accuracy' : overall_speaker_acc,
        'comprehension_accuracy' : overall_listener_acc
    }
    return saved_metrics

def report_hourly_pay(treatment_games, treatment, players):
    print("Reporting hourly pay for the treatment")
    player_to_hourly_pay = {
        player : player_hourly(treatment_games, player) for player in players
    }

    pays = [pay for player, pay in player_to_hourly_pay.items() if player != treatment]
    print(f"The hourly pay for this treatment was: ${np.mean(pays):.2f}/hr")

    return np.mean(pays)

def report_main_metrics(all_games, annotation):
    i2t_path = os.path.join(REFGAME_FOLDER, 'public', 'games', 'idx_to_tangram.json')
    with open(i2t_path) as f:
        idx_to_tangram = json.load(f)

    for treatment in TREATMENTS:
        treatment_games = get_treatment_games(all_games, annotation, treatment)        
        if len(treatment_games) == 0:
            continue
        print(f"Reporting stats for the {len(treatment_games)} {treatment} games collected")

        players = get_list_of_players(treatment_games)
        print(f"{len(players)} distinct players played for this treatment")
        num_players = len(players)

        if treatment == "human":
            language_metrics = report_hh_language_metrics(treatment_games, players) 
            accuracy_metrics = report_hh_accuracy(treatment_games, players, idx_to_tangram) 
        else:
            language_metrics = report_bot_language_metrics(treatment_games, treatment, players)
            accuracy_metrics = report_bot_accuracy(treatment_games, treatment, players, idx_to_tangram)

        hourly_pay = report_hourly_pay(treatment_games, treatment, players)
        print()

        # Save the metrics
        metric_dict = {
            'num_players' : num_players,
            'language_metrics' : language_metrics,
            'accuracy_metrics' : accuracy_metrics,
            'hourly_pay' : hourly_pay
        }
        filename = f'{annotation}_{treatment}.pkl'
        filepath = os.path.join('saved_deployment_metrics', filename)
        with open(filepath, 'wb') as f:
            pickle.dump(metric_dict, f)

def report_suspicious_behavior(all_games, annotation):
    print("Checking for failed attention checks")
    for treatment in TREATMENTS:
        treatment_games = get_treatment_games(all_games, annotation, treatment)        
        if len(treatment_games) == 0 or treatment == "human":
            continue
        
        for game_id, game_dict in treatment_games.items():
            for curr_round in game_dict["roundDicts"]:
                if curr_round["isAttnCheck"] and curr_round["selection"] != curr_round["target"]:
                    print("Failed attention check in: ", game_id)

def report_model_anomalies(all_games, annotation):
    listener_examples = 0
    listener_failures = 0
    listener_idles = 0

    speaker_examples = 0
    speaker_failures = 0
    speaker_idles = 0

    anomalies = []
    for treatment in TREATMENTS:
        treatment_games = get_treatment_games(all_games, annotation, treatment)        
        if len(treatment_games) == 0 or treatment == "human":
            continue

        for game_id, game_dict in treatment_games.items():
            for curr_round in game_dict["roundDicts"]:
                round_index = curr_round["index"]
                if curr_round["listener"] == treatment:
                    if curr_round["chat"] == "Speaker idled":
                        continue

                    listener_examples += 1
                    model_selection = curr_round["selection"]
                    if model_selection is None:
                        print(f"Listener selected None: {(game_id, round_index)}")
                        print()
                        listener_failures += 1
                        anomalies.append((game_id, round_index))
                    elif model_selection == "no_clicks":
                        print(f"Listener idled: {(game_id, round_index)}")                        
                        print()
                        listener_idles += 1                        
                        anomalies.append((game_id, round_index))
                    elif model_selection not in curr_round["listener_context"]:
                        config = game_dict["config"]
                        print(f"Listener failure at: {(game_id, round_index)}. Model selected: {model_selection}. Game config: {config}")
                        print()
                        listener_failures += 1
                        anomalies.append((game_id, round_index))
                else:
                    speaker_examples += 1

                    message = curr_round["chat"]
                    if message == "Speaker failed?":
                        print(f"Speaker did not send a message at: {(game_id, round_index)}")
                        print()
                        speaker_failures += 1                        
                        anomalies.append((game_id, round_index))
                    elif message == "Speaker idled":
                        print(f"Speaker idled at: {(game_id, round_index)}")                        
                        print()
                        speaker_idles += 1                        
                        anomalies.append((game_id, round_index))

                    tokenized_message = tokenize_message(message)
                    if len(tokenized_message) == 0:
                        print(f"Speaker sent an empty message at: {(game_id, round_index)}")                        
                        print()
                        speaker_failures += 1
                        anomalies.append((game_id, round_index))

    print(f"There were model failures in {listener_failures / listener_examples * 100:.2f}% of listener examples")
    print(f"The listener model idled in {listener_idles / listener_examples * 100:.2f}% of examples")

    print(f"There were model failures in {speaker_failures / speaker_examples * 100:.2f}% of speaker examples")
    print(f"The speaker model idled in {speaker_idles / speaker_examples * 100:.2f}% of speaker examples")
                        
    print(anomalies)

## CONTROL ##

def main():
    # Get the args
    args = get_args()
    all_games, all_players = get_all_cached()

    if args.statistic == "data_collected":
        report_data_collected(all_games, args.annotation)
    elif args.statistic == "premature_game_list":
        report_premature_games(all_games, args.annotation)
    elif args.statistic == "main_metrics":
        report_main_metrics(all_games, args.annotation) 
    elif args.statistic == "suspicious_behavior_check":
        report_suspicious_behavior(all_games, args.annotation)
    elif args.statistic == "model_anomaly_check":
        report_model_anomalies(all_games, args.annotation)

if __name__ == "__main__":
    main()
