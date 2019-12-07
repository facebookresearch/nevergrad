#!/bin/bash 

#set -x -e 
# FIXME: Should we include the module load here ?
listxp=`grep -i1 "^def" nevergrad/benchmark/*experiments.py | grep -i1 '@regis' | grep ':def' | sed 's/.*:def //g' | sed 's/(.*//g'`
listxp="hardmultimodal_perf multimodal_perf photonics war preliminary_asynchronous $listxp"

echo working on $listxp
touch allxps
rm -rf allxps
mkdir -p allxps
echo '<html><head><title>nevergrad xps</title></head><body>' > allxps/list.html
echo '<a href="https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/experiments.py"> experiments details</a>'>> allxps/list.html
echo '<a href="https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/frozenexperiments.py"> frozenexperiments details</a>'>> allxps/list.html
for xp in $listxp
do
    echo working on $xp
    echo "<br> $xp </br><p>" >> allxps/list.html
    if [ -d "../outputs/$xp" ]; then
        pushd ..
        python -m dfoptim.benchmark.slurmplot outputs/$xp --max_combsize=2 --competencemaps=True
        mkdir -p nevergrad_repository/allxps/${xp}
        tar -xzf outputs/${xp}.tar.gz ./data.csv
        tar --wildcards -xzf outputs/${xp}.tar.gz ./fight_*.png
        tar --wildcards -xzf outputs/${xp}.tar.gz ./*.tex
        mv fight*.png data.csv *.tex nevergrad_repository/allxps/${xp}/
        gzip -f nevergrad_repository/allxps/${xp}/*.csv
        popd
        echo "<h1> Section ${xp} </h1>" >> allxps/list.html
        ls allxps/${xp}/fight_all.png | sed 's/.*/<img src="https:\/\/dl.fbaipublicfiles.com\/nevergrad\/&">&<p>/g' >> allxps/list.html
        echo '<font size="-6">' >> allxps/list.html
        ls allxps/${xp}/*.png | grep -v fight_all | sed 's/.*/<a href="https:\/\/dl.fbaipublicfiles.com\/nevergrad\/&">&<\/a><br>/g' >> allxps/list.html
        ls allxps/${xp}/*.csv* | sed 's/.*/<a href="https:\/\/dl.fbaipublicfiles.com\/nevergrad\/&">&<\/a>/g' >> allxps/list.html
        ls allxps/${xp}/*.tex* | sed 's/.*/<a href="https:\/\/dl.fbaipublicfiles.com\/nevergrad\/&">&<\/a>/g' >> allxps/list.html
        echo '</font>' >> allxps/list.html
    fi
done
echo '</body></html>' >> allxps/list.html
fs3cmd sync allxps/** s3://dl.fbaipublicfiles.com/nevergrad/allxps/
echo 'https://dl.fbaipublicfiles.com/nevergrad/allxps/list.html'





