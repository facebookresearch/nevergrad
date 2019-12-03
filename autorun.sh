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
    python -m dfoptim.benchmark.slurm $xp --seed=1 --repetitions=7 --num_workers=111 --partition=learnfair  &
    sleep $T
    popd
done
# FIXME: 3600 is not enough for big xps.
sleep $T
sleep $T
sleep $T
sleep $T
echo '<html><head><title>nevergrad xps</title></head><body>' > allxps/list.html
for xp in $listxp
do
    echo "<br> $xp </br><p>" >> allxps/list.html
    pushd ..
    python -m dfoptim.benchmark.slurmplot outputs/$xp --max_combsize=2 
    # FIXME add --competencemaps=True
    mkdir -p nevergrad_repository/allxps/${xp}
    tar -xzf outputs/${xp}.tar.gz ./data.csv
    tar --wildcards -xzf outputs/${xp}.tar.gz ./fight_*.png
    mv fight*.png data.csv nevergrad_repository/allxps/${xp}/
    gzip nevergrad_repository/allxps/${xp}/*.csv
    popd
    echo "<h1> ${xp} <\/h1>" >> allxps/list.html
    ls allxps/${xp}/fight_all.png | sed 's/.*/<img src="https:\/\/dl.fbaipublicfiles.com\/nevergrad\/&">&<p>/g' >> allxps/list.html
    ls allxps/${xp}/*.png | grep -v fight_all | sed 's/.*/<a href="https:\/\/dl.fbaipublicfiles.com\/nevergrad\/&">&<\/a>/g' >> allxps/list.html
    ls allxps/${xp}/*.csv* | sed 's/.*/<a href="https:\/\/dl.fbaipublicfiles.com\/nevergrad\/&">&<\/a>/g' >> allxps/list.html
done
echo '</body></html>' >> allxps/list.html
fs3cmd sync allxps/** s3://dl.fbaipublicfiles.com/nevergrad/allxps/
echo 'https://dl.fbaipublicfiles.com/nevergrad/allxps/list.html'





