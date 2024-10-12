#!/bin/bash
cp multi_ceviche_c0.csv multi_ceviche_c0plot_`date | sed 's/ /_/g'`.csv.back
cp multi_ceviche_c0p.csv multi_ceviche_c0p_plot_`date | sed 's/ /_/g'`.csv.back
cp multi_ceviche_c0_discrete.csv multi_ceviche_c0_discreteplot_`date | sed 's/ /_/g'`.csv.back
grep -v c0c multi_ceviche_c0.csv > multi_ceviche_c0_discrete.csv
grep -v c0c multi_ceviche_c0_warmstart.csv > multi_ceviche_c0_discrete_warmstart.csv
egrep '^loss|CMA|Lengl|Logno' multi_ceviche_c0.csv | grep -v '^[-0-9\.]*,[1-9][,\.]' | grep -v '^[-0-9\.]*,[1-9][0-9][,\.]' | grep -vi c0p | grep -v '^[-0-9\.]*,[1-9][0-9][0-9][,\.]' | grep -v '^[-0-9\.]*,[1-9][0-9][0-9][0-9][,\.]' | grep -vi c0  | grep -iv ng > multi_ceviche_c0_best.csv
touch multi_ceviche_c0_plots
rm -rf multi_ceviche_c0_plots
python -m nevergrad.benchmark.plotting multi_ceviche_c0.csv --max_combsize=1
python -m nevergrad.benchmark.plotting multi_ceviche_c0_discrete.csv --max_combsize=0
python -m nevergrad.benchmark.plotting multi_ceviche_c0_warmstart.csv --max_combsize=0
python -m nevergrad.benchmark.plotting multi_ceviche_c0_discrete_warmstart.csv --max_combsize=0
python -m nevergrad.benchmark.plotting multi_ceviche_c0_best.csv --max_combsize=0
python -m nevergrad.benchmark.plotting multi_ceviche_c0p.csv --max_combsize=0
pushd multi_ceviche_c0p_plots

for u in xpresu*.png
do
cp "$u" ../multi_ceviche_c0_plots/warmup_`echo $u | sed 's/[^0-9a-zA-Z\.]/_/g'`
done
popd
pushd multi_ceviche_c0_best_plots

for u in xpresu*.png
do
cp "$u" ../multi_ceviche_c0_plots/best_`echo $u | sed 's/[^0-9a-zA-Z\.]/_/g'`
done
popd


pushd multi_ceviche_c0_discrete_plots

for u in xpresu*.png
do
cp "$u" ../multi_ceviche_c0_plots/discrete_`echo $u | sed 's/[^0-9a-zA-Z\.]/_/g'`
done
popd

pushd multi_ceviche_c0_discrete_warmstart_plots

for u in xpresu*.png
do
cp "$u" ../multi_ceviche_c0_plots/discrete_warmstart_`echo $u | sed 's/[^0-9a-zA-Z\.]/_/g'`
done
popd

pushd multi_ceviche_c0_warmstart_plots

for u in xpresu*.png
do
cp "$u" ../multi_ceviche_c0_plots/warmstart_`echo $u | sed 's/[^0-9a-zA-Z\.]/_/g'`
done
popd

cat multi_ceviche_c0.csv | sed 's/,[^,]*$//g' | sed 's/.*,//g' | sort | uniq -c | sort -n ; wc -l multi_ceviche_c0.csv 

python examples/convert_ceviche_npy.py

tar -zcvf ~/pixel.tgz LOGPB*.png multi_cev*_plots pb*budget*.png allpngs/*.png histo.png WSpb*budget*.png

echo 'Want to know what BFGS does ?'
grep LOGPB *.out | grep -iv cheating | sed 's/.*://g' | sort | uniq -c | grep with_budget | awk '{ data[$2,"_",$5] += $7; num[$2,"_",$5] += 1  } END { for (u in data) { print u, data[u]/num[u]}   } ' | sort -n | sed 's/_/ /g' | sed 's/[^-LOGPB0-9\.]/ /g'  | sed 's/_/ /g' | sed 's/ [ ]*/ /g' | sort -n -k 2,3 > nocheat.txt
echo "overview ----------------------------------------------"
grep LOGPB *.out | grep -iv cheating | sed 's/.*://g' | sort | uniq -c | grep with_budget | awk '{ data[$2,"_",$5] += $7; num[$2,"_",$5] += 1  } END { for (u in data) { print u, data[u]/num[u]}   } ' | sort -n | sed 's/_/ /g' | sed 's/[^-LOGPB0-9\.]/ /g' | awk '{ data[$1][$2 +0] = $3; datamax[$1] = ( $2 + 0  > datamax[$1] + 0 ? $2 + 0  : datamax[$1] +0 ) } END  { for (pb in data) { print pb, datamax[pb], data[pb][datamax[pb] + 0] } } '
echo 'Want to know what BFGS does when cheating ?'
grep LOGPB *.out | grep -i cheating | sed 's/(//g' | sed 's/, array.*//g' | sed 's/.*://g' | sort | uniq -c | grep with_budget | awk '{ data[$2,"_",$5] += $7; num[$2,"_",$5] += 1  } END { for (u in data) { print u, data[u]/num[u]}   } ' | sort -n | sed 's/_/ /g' | sed 's/[^-LOGPB0-9\.]/ /g' |sed 's/_/ /g' | sed 's/ [ ]*/ /g' | sort -n  -k 1,2 > cheat.txt
echo "overview ----------------------------------------------"
grep LOGPB *.out | grep -i cheating | sed 's/(//g' | sed 's/, array.*//g' | sed 's/.*://g' | sort | uniq -c | grep with_budget | awk '{ data[$2,"_",$5] += $7; num[$2,"_",$5] += 1  } END { for (u in data) { print u, data[u]/num[u]}   } ' | sort -n | sed 's/_/ /g' | sed 's/[^-LOGPB0-9\.]/ /g' | awk '{ data[$1][$2 +0] = $3; datamax[$1] = ( $2 + 0  > datamax[$1] + 0 ? $2 + 0  : datamax[$1] +0 ) } END  { for (pb in data) { print pb, datamax[pb], data[pb][datamax[pb] + 0] } } ' 

echo "Biggest budgets:"
cat multi_ceviche_c0.csv | sed 's/^[0-9\.\-]*,//g' | sed 's/,.*//g' | sort -n | uniq -c | tail -n 10

./scripts/plot_post_ceviche.sh

for k in 0 1 2 3 
do
cp `ls -ctr | grep array | sed 's/_/ /g' | sort -n -r -k4,4 | sed 's/ /_/g' | grep pb$k | tail -n 1` bestnp${k}.npy
done
