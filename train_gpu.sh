#!/bin/bash
#SBATCH --job-name=gputrain
#SBATCH --partition=gpu --gres=gpu:1
##SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err

uv sync --locked
uv run python train.py
