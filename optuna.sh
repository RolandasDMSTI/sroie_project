#!/bin/bash
#SBATCH --job-name=optuna
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=optuna-%j.out
#SBATCH --error=optuna-%j.err

uv sync --locked
uv run python train_with_optuna.py
