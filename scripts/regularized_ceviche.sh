#!/bin/bash
#SBATCH --job-name=regceviche
#SBATCH --output=regceviche_%A_%a.out
#SBATCH --error=regceviche_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH -a 0-50%330

#reg=reg0001
reg=reg10000

# All suffixes below are possibilities:
# multi_ceviche_c0_reg0001
# multi_ceviche_c0_reg001
# multi_ceviche_c0_reg01
# multi_ceviche_c0_reg10
# multi_ceviche_c0_regm1000
# multi_ceviche_c0_reg1000
# multi_ceviche_c0_reg
# multi_ceviche_c0_wsreg0001
# multi_ceviche_c0_wsreg001
# multi_ceviche_c0_wsreg01
# multi_ceviche_c0_wsreg10
# multi_ceviche_c0_wsreg1000
# multi_ceviche_c0_wsregm1000
# multi_ceviche_c0_wsreg

if [ $SLURM_ARRAY_TASK_ID -eq  0 ]; then
cp multi_ceviche_c0_$reg.csv multi_ceviche_c0_${reg}_`date | sed 's/ /_/g'`.csv.back
fi


task=multi_ceviche_c0_$reg
task=multi_ceviche_c0p

echo task attribution $SLURM_ARRAY_TASK_ID $task
echo Keras/TF versions:
pip show keras tensorflow tensorflow-estimator

conda info

echo Starting at
date
# num_workers is the number of processes. Maybe use a bit more than the number of cores at the line "cpus-per-task"
# above.
time python -m nevergrad.benchmark $task --num_workers=1 2>&1 | cut -c1-180 | egrep '[A-Zf-z]'
echo task over $SLURM_ARRAY_TASK_ID $task
echo Finishing at
date
