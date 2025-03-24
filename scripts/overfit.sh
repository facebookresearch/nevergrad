#!/bin/bash
#SBATCH --job-name=overfit
#SBATCH --output=overfit_%A_%a.out
#SBATCH --error=overfit_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=64g
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH -a 0-100



python examples/overfit.py    2>&1 | tail -n 40
