#!/bin/bash 
#SBATCH --job-name=ngrobustoptim
#SBATCH --output=ngrobustoptim_%a_%A.out
#SBATCH --error=ngrobustoptim_%a_%A.err
#SBATCH --time=72:00:00
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH -a 0-100

for a in `seq 120`
do
(
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100
python examples/ng_robustoptim.py | grep seed | cut -c 1-100

) &
done


wait
