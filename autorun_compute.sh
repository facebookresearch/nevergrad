#!/bin/bash 

# See doc in autorun.sh: this script launches the computations, as well as autorun_plot creates the figures.

listxp=`grep -iH1 "^def" nevergrad/benchmark/experiments.py | grep -i1 '@regis' | grep ':def' | sed 's/.*:def //g' | sed 's/(.*//g'`
#listxp="multimodal_perf photonics war preliminary_asynchronous $listxp"

listxp=${1:-$listxp}
touch allxps
rm -rf allxps
T=2000
for xp in $listxp
do  
    pushd ..
	echo "Experiment $xp ==============================="
    # FIXME:
    # Quite small scale for the moment. More repetitions and more workers later, when it will be clear that everything
    # is fine.
    # Using learnfair queue is risky; this will also be discussed
    #rm -rf outputs/${xp}
    python -m dfoptim.benchmark.slurm $xp --seed=1 --time 4320 --repetitions=1 --num_workers=800 --partition=learnfair  &
    sleep $T
    popd
done
