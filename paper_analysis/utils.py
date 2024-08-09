import pickle
import json
import os
import torch
import numpy as np
from tqdm import tqdm

import spacy

BASE_SPLIT_FOLDER = '/home/mog29/cogen/data_and_checkpoints/continual_learning'
SPLIT_FOLDER = '/home/mog29/cogen/data_and_checkpoints/continual_learning/analysis'

def get_and_save_doc_dict(data, doc_filename):
    nlp = spacy.load("en_core_web_sm")
    doc_dict = {}

    # Get the docs
    for game_id, game_dict in tqdm(data.items()):
        if game_id not in doc_dict:
            doc_dict[game_id] = {}
        for round_idx, round_dict in game_dict.items():
            utterance = round_dict['chat'].lower()
            doc = nlp(utterance)
            doc_dict[game_id][round_idx] = doc

    # Save the docs
    with open(doc_filename, 'wb') as f:
        pickle.dump(doc_dict, f)
    return doc_dict

def get_training_treatment_round_json(treatment, deployment_round, suffix):
    data_paths = ['cl_r0.json'] + [f'cl_r{i}_{treatment}_{suffix}.json' for i in range(1, deployment_round)]

    data = {}
    for data_path in data_paths:
        full_path = os.path.join(BASE_SPLIT_FOLDER, data_path)
        with open(full_path, 'r') as f:
            data.update(json.load(f))

    return data

def get_seed_training_docs():
    doc_filename = os.path.join('/home/mog29/cogen', 'paper_analysis',
                                'analysis_cache', 'training_cl_r0_spacy_docs.pkl')
    if os.path.isfile(doc_filename):
        with open(doc_filename, 'rb') as f:
            data = pickle.load(f)
    else:
        seed_path = os.path.join(BASE_SPLIT_FOLDER, 'cl_r0.json')
        with open(seed_path, 'r') as f:
            data = json.load(f)
        data = get_and_save_doc_dict(data, doc_filename)

    new_data = {}
    total = 0
    for game_id, round_dict in data.items():
        if total > 95:
            break
        total += len(round_dict)
        new_data[game_id] = round_dict

    return new_data

def get_training_treatment_round_docs(treatment, deployment_round, suffix):
    data_path = os.path.join(BASE_SPLIT_FOLDER, f'cl_r{deployment_round}_{treatment}_{suffix}.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    doc_filename = os.path.join('/home/mog29/cogen', 'paper_analysis',
                                'analysis_cache', f'training_cl_r{deployment_round}_{treatment}_{suffix}_spacy_docs.pkl')
    if os.path.isfile(doc_filename):
        with open(doc_filename, 'rb') as f:
            doc_dict = pickle.load(f)
    else:
        doc_dict = get_and_save_doc_dict(data, doc_filename)

    # Filter for speaker
    new_doc_dict = {}
    for game_id, game_dict in data.items():
        for round_index, curr_round in game_dict.items():
            bot_role = 'listener' if 'listener' in game_id else 'speaker'
            reward = curr_round['reward']
            if bot_role == 'listener' and reward != 1:
                continue

            if game_id not in new_doc_dict:
                new_doc_dict[game_id] = {}
            new_doc_dict[game_id][round_index] = doc_dict[game_id][round_index]

    return new_doc_dict

def get_training_treatment_round_human_bot_docs(treatment, deployment_round, suffix):
    data_path = os.path.join(BASE_SPLIT_FOLDER, f'cl_r{deployment_round}_{treatment}_{suffix}.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    doc_filename = os.path.join('/home/mog29/cogen', 'paper_analysis',
                                'analysis_cache', f'training_cl_r{deployment_round}_{treatment}_{suffix}_spacy_docs.pkl')
    if os.path.isfile(doc_filename):
        with open(doc_filename, 'rb') as f:
            doc_dict = pickle.load(f)
    else:
        doc_dict = get_and_save_doc_dict(data, doc_filename)

    # Filter for human listener
    new_doc_dict = {}
    for game_id, game_dict in doc_dict.items():
        if 'listener' not in game_id:
            continue
        new_doc_dict[game_id] = game_dict

    return new_doc_dict

def get_generated_treatment_round_docs(treatment, deployment_round, suffix=None):
    if suffix is None:
        doc_filename = os.path.join('/home/mog29/cogen', 'paper_analysis',
                                    'analysis_cache', f'r{deployment_round}_{treatment}_spacy_docs.pkl')
    else:
        doc_filename = os.path.join('/home/mog29/cogen', 'paper_analysis',
                                    'analysis_cache', f'r{deployment_round}_{treatment}_spacy_docs_{suffix}.pkl')
    if os.path.isfile(doc_filename):
        with open(doc_filename, 'rb') as f:
            doc_dict = pickle.load(f)
    else:
        if suffix is None:
            json_path = os.path.join(SPLIT_FOLDER, f"r{deployment_round}_{treatment}.json")
        else:
            json_path = os.path.join(SPLIT_FOLDER, f"r{deployment_round}_{treatment}_{suffix}.json")            
        with open(json_path, 'r') as f:
            data = json.load(f)
        doc_dict = get_and_save_doc_dict(data, doc_filename)

    return doc_dict
        
def get_human_human_round_docs(deployment_round):
    doc_filename = os.path.join('/home/mog29/cogen', 'paper_analysis',
                                'analysis_cache', 'human_human_spacy_docs.pkl')
    with open(doc_filename, 'rb') as f:
        doc_dict = pickle.load(f)

    new_doc_dict = {}
    for game_id, game_dict in doc_dict.items():
        if f"r{deployment_round}" not in game_id:
            continue
        new_doc_dict[game_id] = game_dict

    return new_doc_dict
        
        
    

    
    
