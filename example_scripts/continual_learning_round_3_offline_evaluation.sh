#!/bin/bash
python -m continual_learning.cl_human_model_evals --shared_parameters --evaluation_type=joint --model_family_name=full --deployment_round=2 --eval_dataset=cl_r3_full_all
python -m continual_learning.cl_human_model_evals --shared_parameters --evaluation_type=multitask --model_family_name=no_ji --deployment_round=2 --eval_dataset=cl_r3_no_ji_all
python -m continual_learning.cl_human_model_evals --shared_parameters --evaluation_type=joint --model_family_name=no_ds --deployment_round=2 --eval_dataset=cl_r3_no_ds_listener
python -m continual_learning.cl_human_model_evals --shared_parameters --evaluation_type=multitask --model_family_name=baseline --deployment_round=2 --eval_dataset=cl_r3_baseline_listener


