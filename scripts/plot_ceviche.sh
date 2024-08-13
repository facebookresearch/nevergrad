#!/bin/bash
cp multi_ceviche_c0.csv multi_ceviche_c0plot_`date | sed 's/ /_/g'`.csv.back
touch multi_ceviche_c0_plots
rm -rf multi_ceviche_c0_plots
python -m nevergrad.benchmark.plotting multi_ceviche_c0.csv --max_combsize=1
cat multi_ceviche_c0.csv | sed 's/,[^,]*$//g' | sed 's/.*,//g' | sort | uniq -c | sort -n ; wc -l multi_ceviche_c0.csv 


echo 'Want to know what BFGS does ?'
grep LOGPB *.out | sed 's/.*://g' | sort | uniq -c | grep with_budget | awk '{ data[$2,"_",$5] += $7; num[$2,"_",$5] += 1  } END { for (u in data) { print u, data[u]/num[u]}   } ' | sort -n | grep '800 |1600 '

