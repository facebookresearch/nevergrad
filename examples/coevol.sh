#!/bin/bash 
#SBATCH --job-name=coevol
#SBATCH --output=coevol_%a_%A.out
#SBATCH --error=coevol_%a_%A.err
#SBATCH --time=72:00:00
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH -a 0-80

for a in `seq 80`
do
(
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py
python examples/coevol.py

) &
done


wait
