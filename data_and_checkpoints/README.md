Data and Model Setup Instructions
=================================

Setting Up Data and Models
--------------------------

### Data
To set up data, first download our [collected data](https://huggingface.co/datasets/lil-lab/cogen) from its Huggingface page. The `continual_learning/` and `interaction/` folders should be on the top level of the current folder, while the contents of `refgame_configs/` should be placed under `game_server_side/refgame/public/games`. You should additionally clone the [Kilogram repository](https://github.com/lil-lab/kilogram) and unzip the `square-black-imgs.zip` file inside the `kilogram/dataset` folder.

### Models
To set up models, first create a folder named `experiments/joint_training/` under the current folder. Then, similarly download our [trained models](https://huggingface.co/lil-lab/cogen) from the associated Huggingface page.
All folders defined there should be placed under the `experiments/joint_training/` folder.

### Updating Paths in Configs
You should finally update the absolute paths in `configs/idefics_cl_joint_training.yaml` to match your filepaths.

Description of Data
-------------------

The [Huggingface dataset repository](https://huggingface.co/datasets/lil-lab/cogen) contains three folders:
- The key folder is `continual_learning/`, which contains all of our collected and preprocessed data as well as jsons holding the utterances used in our language analyses.
- The `interaction/` folder contains our collected data without the preprocessing separating datapoints for different systems. That is, each json contains entire interactions between models and individual workers
in Amazon Mechanical Turk HITs.
- The `refgame_configs/` folder contains jsons that the game server loads to serve/render games.