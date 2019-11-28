#!/bin/bash 

# FIXME: Should we include the module load here ?
listxp=`grep -i1 "^def" nevergrad/benchmark/*experiments.py | grep -i1 '@regis' | grep ':def' | sed 's/.*:def //g' | sed 's/(.*//g'`

# FIXME: for this preliminary version, only one xp.
listxp=illcond
for xp in $listxp
do  
    pushd ..
	echo "Experiment $xp ==============================="
    # FIXME:
    # Quite small scale for the moment. More repetitions and more workers later, when it will be clear that everything
    # is fine.
    # Using learnfair queue is risky; this will also be discussed
    python -m dfoptim.benchmark.slurm $xp --seed=1 --repetitions=7 --num_workers=7 --partition=learnfair  &
    popd
done
# FIXME: 3600 is not enough for big xps.
sleep 3600
for xp in $listxp
do
    pushd ..
    python -m dfoptim.benchmark.slurmplot outputs/$xp --max_combsize=2 --competencemaps=True
    mkdir -p nevergrad_repository/allxps/${xp}
    tar -xzf outputs/${xp}.tr.gz data.csv
    tar -xzf outputs/${xp}.tr.gz fight_*.png
    mv fight*.png data.csv nevergrad_repository/allxps/${xp}/
    gzip nevergrad_repository/allxps/${xp}/*
    popd
done
