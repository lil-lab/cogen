#!/bin/bash
python -m continual_learning.cl_precompute_analysis_preds --shared_parameters --evaluation_type=joint --model_family_name=full --deployment_round=2
python -m data_utils.human_human_dataset.create_analysis_model_json --round_idx=2 --treatment=full --sample_suffix=""
python -m continual_learning.cl_precompute_analysis_preds --shared_parameters --evaluation_type=multitask --model_family_name=no_ji --deployment_round=2
python -m data_utils.human_human_dataset.create_analysis_model_json --round_idx=2 --treatment=no_ji --sample_suffix=""
python -m continual_learning.cl_precompute_analysis_preds --shared_parameters --evaluation_type=joint --model_family_name=no_ds --deployment_round=2
python -m data_utils.human_human_dataset.create_analysis_model_json --round_idx=2 --treatment=no_ds --sample_suffix=""
python -m continual_learning.cl_precompute_analysis_preds --shared_parameters --evaluation_type=multitask --model_family_name=baseline --deployment_round=2
python -m data_utils.human_human_dataset.create_analysis_model_json --round_idx=2 --treatment=baseline --sample_suffix=""
