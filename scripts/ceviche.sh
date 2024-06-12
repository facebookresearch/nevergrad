#!/bin/bash
#SBATCH --job-name=ceviche
#SBATCH --output=ceviche_%A_%a.out
#SBATCH --error=ceviche_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH -a 0-250

eval "$(conda shell.bash hook)"
conda activate miniceviche

#task=${tasks[SLURM_ARRAY_TASK_ID]}
if (( RANDOM % 2 )); then
  task=multi_ceviche
else 
  task=multi_ceviche_c0
fi
echo task attribution $SLURM_ARRAY_TASK_ID $task
echo Keras/TF versions:
pip show keras tensorflow tensorflow-estimator

conda info

echo Starting at
date
# num_workers is the number of processes. Maybe use a bit more than the number of cores at the line "cpus-per-task"
# above.
python -m nevergrad.benchmark $task --num_workers=70 2>&1 | tail -n 50
echo task over $SLURM_ARRAY_TASK_ID $task
echo Finishing at
date
