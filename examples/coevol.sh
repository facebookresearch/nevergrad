#!/bin/bash 
#SBATCH --job-name=coevol
#SBATCH --output=coevol_%a_%A.out
#SBATCH --error=coevol_%a_%A.err
#SBATCH --time=72:00:00
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH -a 0-390

for a in `seq 120`
do
(
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py

) | more  &
done


wait
