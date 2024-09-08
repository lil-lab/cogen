CoGen: Learning from Feedback with Coupled Comprehension and Generation
=======================================================================

This is the code repository for the paper "[CoGen: Learning from Feedback with Coupled Comprehension and Generation](https://www.arxiv.org/pdf/2408.15992)," [Mustafa Ömer Gül](https://momergul.github.io/) and [Yoav Artzi](https://yoavartzi.com/) (arXiv preprint).

### About

Systems with both language comprehension and generation capabilities can benefit from the tight connection between the two. This work studies coupling comprehension and generation with focus on continually learning from interaction with users. We propose techniques to tightly integrate the two capabilities for both learning and inference. We situate our studies in two-player reference games, and deploy various models for thousands of interactions with human users, while learning from interaction feedback signals. We show dramatic improvements in performance over time, with comprehension-generation coupling leading to performance improvements up to 26% in absolute terms and up to 17% higher accuracies compared to a non-coupled system. Our analysis also shows coupling has substantial qualitative impact on the system's language, making it significantly more human-like.

Table of Contents
-----------------

- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Setting up the environment](#setting-up-the-environment)
  - [Downloading data and models](#downloading-data-and-models)
- [Training](#training)
  - [Initializing models with seed data](#initializing-models-with-seed-data)
  - [Continual learning training](#continual-learning-training)
- [Evaluation](#evaluation)
  - [Computing offline metrics](#computing-offline-metrics)
  - [Generating utterances](#generating-utterances)
  - [Running language analyses](#running-language-analyses)
- [Model Serving](#model-serving)

Repository Structure
--------------------

- `continual_learning/` contains scripts for training and evaluating models as well as preprocessing utilities used in-between continual learning rounds.
- `models/` contains model definitions and utilities for model loading/saving and optimization
- `data_utils/` contains dataloader utilities used in training as well as data processing scripts used in-between continual learning rounds.
- `bots/` contains scripts used to serve models during deployment
- `configs/` contains .yaml files holding default settings for training/evaluation and for model serving
- `utils/` contains utility functions used across multiple subfolders
- `game_server_side/` contains all code for the game server and for generating reference games contexts used in deployment
- `data_and_checkpoints/` contains the data collected in deployment and trained model checkpoints
- `visualization_scripts` contains streamlit visualization scripts for debugging or inspecting collected data
- `paper_analysis/` contains the scripts used to compute results
- `example_scripts/` contains example scripts for model training and evaluation
- `similariy_files/` contains the CLIP similarity scores used to construct games

Installation
------------

### Setting up the environment

1. To set up the environment, first create a Conda virtual environment and install PyTorch through conda:
```
conda create -n cogen python=3.10
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

2. Install the remaining requirements through pip to finish the environment setup:
```
pip install -r requirements.txt
```

### Downloading data and models

To populate the `data_and_checkpoints/` folder with the collected data and model checkpoints, follow the instructions on `data_and_checkpoints/README.md`

Training
--------

### Initializing models with seed data

For the first round of deployment, models are trained on a small seed dataset of human-human games. To train the Full system on seed data, run the following command:

```
python -m continual_learning.cl_joint_training --shared_parameters --evaluation_type=joint --model_family_name=full --deployment_round=0 --wandb_experiment_name=r0_full --from_scratch --use_wandb --only_seed
```

To similarly train the No-JI baseline on seed data, run the following:
```
python -m continual_learning.cl_joint_training --shared_parameters --evaluation_type=multitask --model_family_name=no_ji --deployment_round=0 --wandb_experiment_name=r0_no_ji --from_scratch --use_wandb --only_seed
```

Here, the `evaluation_type` argument controls whether joint inference is performed during evaluation, with `joint` and `multitask` indicating the use and lack of joint inference respectively. You may use the `name_suffix` argument to specify a name for your training experiment and avoid overwriting our provided checkpoints.

### Continual learning training

For continual learning training, the main added consideration is the use of data sharing. To train the Full system on data collected up to round <round>, run the following command:

```
python -m continual_learning.cl_joint_training --shared_parameters --evaluation_type=joint --model_family_name=full --deployment_round=<round> --wandb_experiment_name=r<round>_full --from_scratch --use_wandb --use_separate_dataloaders --listener_filter=no_neg_gen  --speaker_filter=no_neg_comp
```

Here, the `use_separate_dataloaders` argument specifies that separate dataloaders should be used for comprehension and generation tasks, with the `listener_filter` and `speaker_filter` controlling what datapoints are shared across tasks during data sharing. If data sharing is not used, these arguments are not included in the command. Example scripts for all other systems can be found in the `example_scripts` folder.

Evaluation
----------

### Computing offline metrics

The `continual_learning/cl_human_model_evals.py` script can be used to evaluate model checkpoints on offline metrics such as comprehension accuracy. To evaluate the Full system deployed on the second round on interactions it collected, for instance, run the following:

```
python -m continual_learning.cl_human_model_evals --shared_parameters --evaluation_type=joint --model_family_name=full --deployment_round=1 --eval_dataset=cl_r2_full_all
```

Example scripts for evaluating other systems can be found in the `example_scripts` folder.

### Generating utterances

To prepare utterances to be used in language analyses, there are two steps:

1. First use the `continual_learning/cl_precompute_analysis_preds.py` script to generate utterances. For the Full system checkpoint deployed on the second round, for instance, run:

```
python -m continual_learning.cl_precompute_analysis_preds --shared_parameters --evaluation_type=joint --model_family_name=full --deployment_round=1
```

2. Next, place the generated utterances into a structured json format (to be then used in scripts under `paper_analysis`). Yet again, for the Full system on the second round, run:

```
python -m data_utils.human_human_dataset.create_analysis_model_json --round_idx=2 --treatment=full --sample_suffix=""
```

The `sample_suffix` argument can be changed to avoid overwriting our provided data.

### Running language analyses

To run language analyses after generating utterances for all system-round pairs, simply execute the cells on `paper_analysis/Main Language Results.ipynb`. This script will compute results over a single sub-sample for each system-round pair. To obtain a better estimate over multiple subsamples:

1. Precompute GPT-2 embeddings for each system-round pair by running the following while in `paper_analysis`:
```
python precompute_gpt2_embeds.py
```

2. Compute and save language analysis results over multiple sub-samples:
```
python multi_sample_language_analyses.py --num_samples=10000 --sample_id=0
```

3. Run the cells in `paper_analysis/Main Language Results Multiple Samples.ipynb`

Model Serving
-------------
During deployment, the game server directs requests to a Tornado app routing requests to the relevant models. To run the app, execute the following command:

```
python -m bots.main --configuration_file=r0_bot_serving.yaml --num_gpus=3
```

Here, the `configuration_file` argument points to a .yaml file held in the `configs/bot_configs/experiment_configs` directory. This .yaml file itself contains a list of config yamls (found in `configs/bot_configs/manager_configs`), each of which specifies what model to serve and with what inference details. Each model instance being served has a GPU dedicated to it and the `num_gpus` argument should match the number of model instances.