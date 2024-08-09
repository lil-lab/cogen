import numpy as np
import os
import pickle
import json
import matplotlib.pyplot as plt
from surface_level_analyses import LexicalSurfaceLevelAnalysis
import random
import torch
from mauve import compute_mauve
from tqdm import tqdm
import argparse

## Global variables
ANALYSIS_PATH = '/home/mog29/cogen/data_and_checkpoints/continual_learning/analysis/human_human.json'
EMBED_FOLDER = '/home/mog29/cogen/paper_analysis/analysis_cache/gpt2_features/'
CACHE_PATH = '/home/mog29/cogen/paper_analysis/analysis_cache/intermediate_language_computations'
with open(ANALYSIS_PATH, 'r') as f:
    human_human_datapoints = json.load(f)

generation_analysis = LexicalSurfaceLevelAnalysis("generation")
TREATMENTS = ['full', 'no_ji', 'no_ds', 'baseline']
for treatment in TREATMENTS:
    start_i = 1 if treatment in ['full', 'no_ji'] else 2
    for i in range(start_i, 5):
        print(treatment, i)
        generation_analysis.add_utterances(treatment, i)

human_human_analysis = LexicalSurfaceLevelAnalysis("human")
for i in range(1, 5):
    print('human', i)
    human_human_analysis.add_utterances('human', i)

## Analysis functions

def sample_subsets():
    round_to_subsets = {}

    for i in range(1, 5):
        game_round_pairs = []
        for game_id, game_dict in human_human_datapoints.items():
            if f'r{i}' not in game_id:
                continue
            for round_idx, round_dict in game_dict.items():
                game_round_pairs.append((game_id, round_idx))
        random.shuffle(game_round_pairs)
        subsampled_pairs = game_round_pairs[:1949]
        round_to_subsets[i] = subsampled_pairs

    return round_to_subsets

def get_utterance_length(treatment, deployment_round, game_subsets):
    game_subset = game_subsets[deployment_round]
    if treatment == "human":
        _, utt_length = human_human_analysis.utterance_length(treatment, deployment_round, game_subset=game_subset)
    else:
        _, utt_length = generation_analysis.utterance_length(treatment, deployment_round, game_subset=game_subset)
    return utt_length

def get_vocab_size(treatment, deployment_round, game_subsets):
    game_subset = game_subsets[deployment_round]
    if treatment == "human":
        _, vocab_size = human_human_analysis.vocabulary_size(treatment, deployment_round, game_subset=game_subset)
    else:
        _, vocab_size = generation_analysis.vocabulary_size(treatment, deployment_round, game_subset=game_subset)
    return vocab_size

def get_new_words(treatment, deployment_round, game_subsets):
    seen_words = set()
    for i in range(1, deployment_round):
        game_subset = game_subsets[i]

        curr_treatment = treatment
        if i == 1 and treatment in ['no_ds', 'baseline']:
            curr_treatment = 'full' if treatment == 'no_ds' else 'no_ji'
        generated_vocab, _ = generation_analysis.vocabulary_size(curr_treatment, i, game_subset=game_subset)

        for word in generated_vocab:
            seen_words.add(word)

    game_subset = game_subsets[deployment_round]
    generated_vocab, _ = generation_analysis.vocabulary_size(treatment, deployment_round, game_subset=game_subset)

    new_words = [word for word in generated_vocab if word not in seen_words]
    return len(new_words)

def get_specific_embeddings(treatment, deployment_round, game_subset):
    path = os.path.join(EMBED_FOLDER, f'r{deployment_round}_{treatment}.pth')
    all_embeds = torch.load(path)

    embeds = []
    for game_id, game_dict in all_embeds.items():
        for round_idx, round_embedding in game_dict.items():
            if (game_id, round_idx) not in game_subset:
                continue
            embeds.append(round_embedding)

    return np.stack(embeds, axis=0)

def get_subsample_mauve(treatment, deployment_round, game_subsets):
    game_subset = game_subsets[deployment_round]
    model_embeddings = get_specific_embeddings(treatment, deployment_round, game_subset)
    human_embeddings = get_specific_embeddings('human', deployment_round, game_subset)
    return compute_mauve(p_features=human_embeddings, q_features=model_embeddings, num_buckets=200).mauve

def single_sample_analysis(treatment, deployment_round):
    game_subsets = sample_subsets()

    if treatment != "human":
        utterance_length = get_utterance_length(treatment, deployment_round, game_subsets)
        vocab_size = get_vocab_size(treatment, deployment_round, game_subsets)
        new_words = get_new_words(treatment, deployment_round, game_subsets)
        mauve = get_subsample_mauve(treatment, deployment_round, game_subsets)
        return utterance_length, vocab_size, new_words, mauve
    else:
        utterance_length = get_utterance_length(treatment, deployment_round, game_subsets)
        vocab_size = get_vocab_size(treatment, deployment_round, game_subsets)
        return utterance_length, vocab_size

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--sample_id', type=int)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    num_samples = args.num_samples
    sample_id = args.sample_id

    treatment_to_results = {}
    for treatment in ['full', 'no_ji', 'no_ds', 'baseline', 'human']:
        treatment_to_results[treatment] = {}
        start_idx = 2 if treatment in ['no_ds', 'baseline'] else 1
        for deployment_round in range(start_idx, 5):
            treatment_to_results[treatment][deployment_round] = []
            for i in tqdm(range(num_samples)):
                outputs = single_sample_analysis(treatment, deployment_round)
                treatment_to_results[treatment][deployment_round].append(outputs)

    savepath = os.path.join(CACHE_PATH, f'intermediate_{sample_id}.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(treatment_to_results, f)

if __name__ == "__main__":
    main()
    

    
