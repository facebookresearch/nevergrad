#!/bin/bash 
#SBATCH --job-name=ngcoevol
#SBATCH --output=ngcoevol_%a_%A.out
#SBATCH --error=ngcoevol_%a_%A.err
#SBATCH --time=72:00:00
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH -a 0-100

for a in `seq 120`
do
(
(
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more
python examples/ng_coevol.py | grep seed | more

) | more 
)&
done


wait
