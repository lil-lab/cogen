#!/bin/bash
python -m continual_learning.cl_joint_training --shared_parameters --evaluation_type=joint --model_family_name=full --deployment_round=0 --wandb_experiment_name=r0_full --from_scratch --use_wandb --only_seed --name_suffix=<exp_name>
python -m continual_learning.cl_joint_training --shared_parameters --evaluation_type=multitask --model_family_name=no_ji --deployment_round=0 --wandb_experiment_name=r0_no_ji --from_scratch --use_wandb --only_seed --name_suffix=<exp_name>
