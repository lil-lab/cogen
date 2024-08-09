# File: surface_level_analyses.py
# -------------------------------
# Helper script to compute surface level features.

import pickle
import json
import os
import torch
import numpy as np
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from transformers import AutoProcessor
from collections import Counter
import spacy

from utils import get_seed_training_docs, get_training_treatment_round_docs, get_generated_treatment_round_docs, \
    get_training_treatment_round_json, get_training_treatment_round_human_bot_docs, get_human_human_round_docs

BASE_SPLIT_FOLDER = '/home/mog29/cogen/data_and_checkpoints/continual_learning'
SPLIT_FOLDER = '/home/mog29/cogen/data_and_checkpoints/continual_learning/analysis'
SUBSET_PATH = '/home/mog29/cogen/paper_analysis/analysis_cache/game_subsets'

def dependency_metric(doc):
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
    
def n_gram_diversity(data, analysis_level, hh_round):
    # First initialize the tokenizer
    if analysis_level == "subword":
        checkpoint = "HuggingFaceM4/idefics2-8b"
        tokenizer = AutoProcessor.from_pretrained(checkpoint, do_image_splitting=False,
                                                  size={"longest_edge": 448, "shortest_edge": 224})

    values = []
    for n in range(2, 5):
        unique_grams = set()
        total_grams = 0

        for game_id, game_dict in tqdm(data.items()):
            curr_round_string = game_id.split("_")[0]
            if hh_round is not None and hh_round != curr_round_string:
                continue

            for round_idx, round_dict in game_dict.items():
                if analysis_level == "word":
                    tokens = [token.text for token in round_dict]
                else:
                    utterance = round_dict['chat'].lower()
                    tokens = tokenizer(utterance)['input_ids'][0][1:]
                if len(tokens) < n:
                    continue

                for i in range(len(tokens) - (n-1)):
                    n_gram = tuple(tokens[i:i+n])
                    unique_grams.add(n_gram)
                    total_grams += 1

        values.append(len(unique_grams) / total_grams)

    metric = 1
    for value in values:
        metric *= value
    values.append(metric)

    return values

def compute_lexical_diversity_statistics_for_system(system_name, hh_round=None):
    # Get the spacy processed language
    print("Loading spacy")
    doc_filename = os.path.join('/home/mog29/cogen', 'paper_analysis', 'analysis_cache', f'{system_name}_spacy_docs.pkl')
    if os.path.isfile(doc_filename):
        with open(doc_filename, 'rb') as f:
            doc_dict = pickle.load(f)
    else:
        doc_dict = get_and_save_doc_dict(data, doc_filename)

    # Load the json for the system
    json_path = os.path.join(SPLIT_FOLDER, f"{system_name}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Compute ratios for different n-s
    wl_outputs = n_gram_diversity(doc_dict, "word", hh_round)
    swl_outputs = n_gram_diversity(data, "subword", hh_round)

    return wl_outputs, swl_outputs

class LexicalSurfaceLevelAnalysis:

    def __init__(self, data_source):
        self.data_source = data_source
        self.treatment_to_utterances = {}

    def add_utterances(self, treatment, round_idx, suffix=None, human_bot=False):
        utterance_dict = {}
        if self.data_source == "training" and treatment != "human":
            utterance_dict.update(get_seed_training_docs())
            for i in range(1, round_idx):
                suffix = 'all' if treatment in ['full', 'no_ji'] else 'speaker'
                utterance_dict.update(get_training_treatment_round_docs(treatment, i, suffix))
        elif self.data_source == "human" and human_bot:
            suffix = 'all' if treatment in ['full', 'no_ji'] else 'listener'
            utterance_dict.update(get_training_treatment_round_human_bot_docs(treatment, round_idx, suffix))
        elif self.data_source == "human" and not human_bot:
            if round_idx == 0:
                utterance_dict.update(get_seed_training_docs())
            else:
                utterance_dict.update(get_human_human_round_docs(round_idx))            
        else:
            utterance_dict.update(get_generated_treatment_round_docs(treatment, round_idx, suffix=suffix))

        if treatment not in self.treatment_to_utterances:
            self.treatment_to_utterances[treatment] = {}
        if round_idx not in self.treatment_to_utterances[treatment]:
            self.treatment_to_utterances[treatment][round_idx] = []
        self.treatment_to_utterances[treatment][round_idx].append(utterance_dict)

    def get_round_utterances(self, treatment, deployment_round, human_human=False):
        '''
        Function restricted to the training set. Returns processed utterances for the desired
        round, coupled with rewards for said round.
        '''
        new_utterance_dict = {}
        if self.data_source == "training":
            suffix = 'all' if treatment in ['full', 'no_ji'] else 'speaker'
            train_json = get_training_treatment_round_json(treatment, deployment_round + 1, suffix)
            utterance_dict = self.treatment_to_utterances[treatment][deployment_round + 1][0]
        elif self.data_source == "human" and human_human:
            train_path = '/home/mog29/cogen/data_and_checkpoints/continual_learning/analysis'
            with open(os.path.join(train_path, 'human_human.json'), 'r') as f:
                train_json = json.load(f)
            utterance_dict = self.treatment_to_utterances[treatment][deployment_round][0]            
        else:
            suffix = 'all' if treatment in ['full', 'no_ji'] else 'listener'
            train_json = get_training_treatment_round_json(treatment, deployment_round + 1, suffix)
            utterance_dict = self.treatment_to_utterances[treatment][deployment_round][0]            
            
        role = "speaker" if self.data_source == "training" else "listener"
        for game_id, game_dict in utterance_dict.items():
            if f"r{deployment_round}" not in game_id:
                continue
            if not human_human and role not in game_id:
                continue
            new_utterance_dict[game_id] = {}

            for round_idx, proc_doc in game_dict.items():
                # Get the reward
                og_dict = train_json[game_id][round_idx]
                reward = og_dict['reward'] if 'reward' in og_dict else 1
                new_utterance_dict[game_id][round_idx] = (proc_doc, reward)

        return new_utterance_dict

    def vocabulary_size(self, treatment, deployment_round, aggregate=False, speaker_only=False, 
                        use_subset=False, game_subset=None):
        if use_subset:
            subset_path = os.path.join(SUBSET_PATH, f'round_{deployment_round}_subsets_alt.pkl')
            with open(subset_path, 'rb') as f:
                game_subset = pickle.load(f)

        vocabulary = set()
        utterance_dicts = self.treatment_to_utterances[treatment][deployment_round]
        indices = [0] if not aggregate else list(range(len(utterance_dicts)))
        for idx in indices:
            utterance_dict = utterance_dicts[idx]
            for game_id, game_dict in utterance_dict.items():
                if speaker_only and (f'r{deployment_round-1}' in game_id and 'listener' in game_id):
                    continue
                for round_idx, proc_doc in game_dict.items():
                    if game_subset is not None and (game_id, round_idx) not in game_subset:
                        continue
                    for token in proc_doc:
                        vocabulary.add(token.text)

        return vocabulary, len(vocabulary)

    def generation_vocabulary_size_from_round(self, treatment, round_idx, filter_round):
        vocabulary = set()
        utterance_dict = self.treatment_to_utterances[treatment][round_idx][0]
        for game_id, game_dict in utterance_dict.items():
            if f'r{filter_round}' not in game_id:
                continue
            for round_idx, proc_doc in game_dict.items():
                for token in proc_doc:
                    vocabulary.add(token.text)

        return len(vocabulary)

    def vocabulary_count(self, treatment, round_idx, aggregate=False, use_subset=False):
        if use_subset:
            subset_path = os.path.join(SUBSET_PATH, f'round_{round_idx}_subsets.pkl')
            with open(subset_path, 'rb') as f:
                game_subset = pickle.load(f)

        vocabulary_counter = Counter()
        utterance_dicts = self.treatment_to_utterances[treatment][round_idx]
        indices = [0] if not aggregate else list(range(len(utterance_dicts)))
        for idx in indices:
            utterance_dict = utterance_dicts[idx]
            for game_id, game_dict in utterance_dict.items():
                if use_subset and game_id not in game_subset:
                    continue
                for round_idx, proc_doc in game_dict.items():
                    for token in proc_doc:
                        vocabulary_counter[token.text] += 1

        return vocabulary_counter

    def vocabulary_reward_count(self, treatment, round_idx):
        '''
        Function restricted to the training set.
        '''
        assert(self.data_source == "training")
        suffix = 'all' if treatment in ['full', 'no_ji'] else 'speaker'
        train_json = get_training_treatment_round_json(treatment, round_idx, suffix)

        vocab_counter = {}
        utterance_dict = self.treatment_to_utterances[treatment][round_idx][0]
        for game_id, game_dict in utterance_dict.items():
            for round_idx, proc_doc in game_dict.items():
                # Get the reward
                og_dict = train_json[game_id][round_idx]
                reward = og_dict['reward'] if 'reward' in og_dict else 1
                    
                for token in proc_doc:
                    token_text = token.text
                    if token_text not in vocab_counter:
                        vocab_counter[token_text] = Counter()
                    vocab_counter[token_text][reward] += 1

        return vocab_counter

    def fine_grained_training_vocab(self, treatment, deployment_round):
        assert(self.data_source == "training")        
        suffix = 'all' if treatment in ['full', 'no_ji'] else 'speaker'
        train_json = get_training_treatment_round_json(treatment, deployment_round, suffix)

        vocab_stat_dict = {}
        utterance_dict = self.treatment_to_utterances[treatment][deployment_round][0]
        for game_id, game_dict in utterance_dict.items():
            utterer = 'human' if 'speaker' not in game_id else 'model'
            curr_round = 0
            for candidate_round in range(1, deployment_round):
                if f'r{candidate_round}' in game_id:
                    curr_round = candidate_round

            for round_idx, proc_doc in game_dict.items():            
                # Get the reward
                og_dict = train_json[game_id][round_idx]
                reward = og_dict['reward'] if 'reward' in og_dict else 1

                for token in proc_doc:
                    curr_word = token.text
                    if curr_word not in vocab_stat_dict:
                        vocab_stat_dict[curr_word] = {}
                    if curr_round not in vocab_stat_dict[curr_word]:
                        vocab_stat_dict[curr_word][curr_round] = {}
                    if utterer not in vocab_stat_dict[curr_word][curr_round]:
                        vocab_stat_dict[curr_word][curr_round][utterer] = {1 : 0, -1: 0}
                    vocab_stat_dict[curr_word][curr_round][utterer][reward] += 1

        return vocab_stat_dict

    def get_human_model_round_vocab(self, treatment, deployment_round):
        '''
        Get the model utterances from round immediately preceding (the final round 
        represented in the training set).
        '''
        vocab = set()
        utterance_dict = self.treatment_to_utterances[treatment][deployment_round][0]
        for game_id, game_dict in utterance_dict.items():
            if f'r{deployment_round-1}' not in game_id or 'speaker' not in game_id:
                continue
            for round_idx, proc_doc in game_dict.items():
                for token in proc_doc:
                    vocab.add(token.text)
        return vocab

    def utterance_length(self, treatment, round_idx, use_subset=False, aggregate=False,
                        game_subset=None):
        if use_subset:
            subset_path = os.path.join(SUBSET_PATH, f'round_{round_idx}_subsets_alt.pkl')
            with open(subset_path, 'rb') as f:
                game_subset = pickle.load(f)

        lengths = []
        utterance_dicts = self.treatment_to_utterances[treatment][round_idx]
        indices = [0] if not aggregate else list(range(len(utterance_dicts)))
        for idx in indices:
            utterance_dict = utterance_dicts[idx]
            for game_id, game_dict in utterance_dict.items():
                for round_idx, proc_doc in game_dict.items():
                    if game_subset is not None and (game_id, round_idx) not in game_subset:
                        continue
                    lengths.append(len(proc_doc))

        return lengths, np.mean(lengths)

    def dependency_metric(self, treatment, round_idx, aggregate=False, use_subset=False):
        if use_subset:
            subset_path = os.path.join(SUBSET_PATH, f'round_{round_idx}_subsets.pkl')
            with open(subset_path, 'rb') as f:
                game_subset = pickle.load(f)

        max_widths = []
        average_branchings = []
        max_depths = []
        utterance_dicts = self.treatment_to_utterances[treatment][round_idx]
        indices = [0] if not aggregate else list(range(len(utterance_dicts)))
        for idx in indices:
            utterance_dict = utterance_dicts[idx]
            for game_id, game_dict in utterance_dict.items():
                if use_subset and game_id not in game_subset:
                    continue
                for round_idx, proc_doc in game_dict.items():
                    max_depth, max_width, average_branching = dependency_metric(proc_doc)
                    max_widths.append(max_width)
                    max_depths.append(max_depth)
                    average_branchings.append(average_branching)

        return np.mean(max_widths), np.mean(average_branchings), np.mean(max_depths)

    def part_of_speech(self, treatment, round_idx, aggregate=False):
        pos_counter = Counter()
        pos_to_vocab = {}

        utterance_dicts = self.treatment_to_utterances[treatment][round_idx]
        indices = [0] if not aggregate else list(range(len(utterance_dicts)))
        for idx in indices:
            utterance_dict = utterance_dicts[idx]
            for game_id, game_dict in utterance_dict.items():
                for round_idx, proc_doc in game_dict.items():
                    for token in proc_doc:
                        pos_counter[token.pos_] += 1
                        if token.pos_ not in pos_to_vocab:
                            pos_to_vocab[token.pos_] = set()
                        pos_to_vocab[token.pos_].add(token.text)

        total = sum([value for _, value in pos_counter.items()])
        for key in pos_counter:
            pos_counter[key] = pos_counter[key] / total

        total_count = sum([len(pos_vocab) for _, pos_vocab in pos_to_vocab.items()])
        pos_to_count = {pos : len(pos_vocab) for pos, pos_vocab in pos_to_vocab.items()}
        pos_to_proportion = {pos : len(pos_vocab) / total_count for pos, pos_vocab in pos_to_vocab.items()}

        return pos_counter, pos_to_proportion, pos_to_count, pos_to_vocab
        
        
                    
                    
        
                    
        
        
