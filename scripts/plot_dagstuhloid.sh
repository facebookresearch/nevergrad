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
    #python -m nevergrad.benchmark.plotting --max_combsize=1  $i 
    (python -m nevergrad.benchmark.plotting --nomanyxp=1 $i ; 
     python -m nevergrad.benchmark.plotting --max_combsize=1 --competencemaps=0 --nomanyxp=1 $i ;
     python -m nevergrad.benchmark.plotting --max_combsize=0 --competencemaps=0 --nomanyxp=0 $i ) > log_${i}.log &
done
wait


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

python -m nevergrad.benchmark.plotting --max_combsize=2 --competencemaps=1 yabigbbob.csv

tar -zcvf dagstuhloid.tgz dagstuhloid.pdf *.csv *plots/xpresults_all.png rnk_*.txt *plots/fight_all.png.cp.txt 

tar -zcvf ~/lamamd.tgz `ls *plots/xpresults*.png  | grep -v ','` `ls *plots/fight_*pure.png | grep -v ',.*,.*,'` dagstuhloid.tex dagstuhloid.pdf */competencemap_dimension,budget.pdf
ls -ctrl *plots/fight_all_pure.png

tar -zcvf ~/yabigbboblama.tgz yabigbbob_plots/xpresul*.png yabigbbob_plots/fight_*.png
