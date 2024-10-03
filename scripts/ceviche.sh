#!/bin/bash
#SBATCH --job-name=ceviche
#SBATCH --output=ceviche_%A_%a.out
#SBATCH --error=ceviche_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH -a 0-300%330





if [ $SLURM_ARRAY_TASK_ID -eq  0 ]; then
cp multi_ceviche_c0.csv multi_ceviche_c0_`date | sed 's/ /_/g'`.csv.back
fi


task=multi_ceviche_c0

echo task attribution $SLURM_ARRAY_TASK_ID $task
echo Keras/TF versions:
pip show keras tensorflow tensorflow-estimator

conda info

echo Starting at
date
# num_workers is the number of processes. Maybe use a bit more than the number of cores at the line "cpus-per-task"
# above.
time python -m nevergrad.benchmark $task --num_workers=1 2>&1 | cut -c1-180 | egrep '[A-Zf-z]'
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#if ! ((  $SLURM_ARRAY_TASK_ID % 30 )) then
#cp multi_ceviche_c0.csv multi_ceviche_c0_`date | sed 's/ /_/g'`.csv.back
#fi
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#if [ $SLURM_ARRAY_TASK_ID -eq  0 ]; then
#cp multi_ceviche_c0.csv multi_ceviche_c0_`date | sed 's/ /_/g'`.csv.back
#fi
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#if [ $SLURM_ARRAY_TASK_ID -eq  0 ]; then
#cp multi_ceviche_c0.csv multi_ceviche_c0_`date | sed 's/ /_/g'`.csv.back
#fi
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
#python -m nevergrad.benchmark $task --num_workers=1 2>&1 | tail -n 50
echo task over $SLURM_ARRAY_TASK_ID $task
echo Finishing at
date
