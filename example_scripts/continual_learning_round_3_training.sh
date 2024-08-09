#!/bin/bash
python -m continual_learning.cl_joint_training --shared_parameters --evaluation_type=joint --model_family_name=full --deployment_round=2 --wandb_experiment_name=r2_full --from_scratch --use_wandb --use_separate_dataloaders --listener_filter=no_neg_gen  --speaker_filter=no_neg_comp
python -m continual_learning.cl_joint_training --shared_parameters --evaluation_type=multitask --model_family_name=no_ji --deployment_round=2 --wandb_experiment_name=r2_no_ji --from_scratch --use_wandb --use_separate_dataloaders --listener_filter=no_neg_gen  --speaker_filter=no_neg_comp
python -m continual_learning.cl_joint_training --shared_parameters --evaluation_type=joint --model_family_name=no_ds --deployment_round=2 --wandb_experiment_name=r2_no_ds --from_scratch --use_wandb
python -m continual_learning.cl_joint_training --shared_parameters --evaluation_type=multitask --model_family_name=baseline --deployment_round=2 --wandb_experiment_name=r2_baseline --from_scratch --use_wandb

