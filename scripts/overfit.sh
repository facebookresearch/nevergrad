#!/bin/bash
#SBATCH --job-name=overfit
#SBATCH --output=overfit_%A_%a.out
#SBATCH --error=overfit_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH -a 0-1



python examples/overfit.py
