#!/bin/bash
#SBATCH --job-name=cputrain
#SBATCH --partition=main
##SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err

uv sync --locked
uv run python train.py
