#!/bin/bash 

# FIXME: Should we include the module load here ?
listxp=`grep -i1 "^def" nevergrad/benchmark/*experiments.py | grep -i1 '@regis' | grep ':def' | sed 's/.*:def //g' | sed 's/(.*//g'`

touch allxps
rm -rf allxps
T=3600
for xp in $listxp
do  
    pushd ..
	echo "Experiment $xp ==============================="
    # FIXME:
    # Quite small scale for the moment. More repetitions and more workers later, when it will be clear that everything
    # is fine.
    # Using learnfair queue is risky; this will also be discussed
    rm -rf outputs/${xp}
    python -m dfoptim.benchmark.slurm $xp --seed=1 --repetitions=7 --num_workers=111 --partition=learnfair  &
    sleep $T
    popd
done
# FIXME: 3600 is not enough for big xps.
sleep $T
sleep $T
sleep $T
sleep $T
./autorun_plot.sh
