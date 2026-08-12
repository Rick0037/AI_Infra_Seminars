#!/bin/bash
#SBATCH --job-name=pytorch_train
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# cd /home/lcpu/46137424/ruixuan
# uv run python src/ruixuan/learning_pytorch.py
uv run python src/ruixuan/module.py
