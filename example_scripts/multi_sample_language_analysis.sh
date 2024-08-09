#!/bin/bash
python precompute_gpt2_embeds.py
python multi_sample_language_analyses.py --num_samples=10000 --sample_id=0
