#!/bin/bash
cp multi_ceviche_c0.csv multi_ceviche_c0plot_`date | sed 's/ /_/g'`.csv.back
touch multi_ceviche_c0_plots
rm -rf multi_ceviche_c0_plots
python -m nevergrad.benchmark.plotting multi_ceviche_c0.csv --max_combsize=1
cat multi_ceviche_c0.csv | sed 's/,[^,]*$//g' | sed 's/.*,//g' | sort | uniq -c | sort -n ; wc -l multi_ceviche_c0.csv 

