#!/bin/bash 
#SBATCH --job-name=dagplot
#SBATCH --output=dagplot.out
#SBATCH --error=dagplot.err
#SBATCH --time=72:00:00
#SBATCH --partition=devlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=67



# This script works both as a standalone script or with slurm.
# This is much slower than mini_plot_dagstuhloid, but outputs more data (in particular: competence maps and plots for
# subcases).
# This plots the results which are stored in the CSV files.

# Do nothing if there is no CSV.
if compgen -G "*.csv" > /dev/null; then

# First we run all nevergrad plotting.
for i in `ls *.csv `
do
    #python -m nevergrad.benchmark.plotting --nomanyxp=1 $i &
    (python -m nevergrad.benchmark.plotting --nomanyxp=1 $i ; python -m nevergrad.benchmark.plotting --max_combsize=2 --competencemaps=1 --nomanyxp=1 $i ) &

done
wait

# ./scripts/multirank.sh
# ./scripts/multirank_robust.sh
./scripts/pies_multirank.sh 
./scripts/get_statistics.sh

# Second we do pdflatex
for i in *.csv
do
    pushd `echo $i | sed 's/\.csv/_plots/g'`
    if compgen -G "comp*.tex" > /dev/null; then
    for t in comp*.tex
    do
        pdflatex $t &
    done
    fi  # end of "there are competence map files"
    popd
done
wait

fi # End of "there is something to do".

# tar -zcvf ~/dag.tgz *_plots
scripts/latexize.sh

tar -zcvf dagstuhloid.tgz dagstuhloid.pdf *.csv *plots/xpresults_all.png rnk_*.txt *plots/fight_all.png.cp.txt *pie*.png 


