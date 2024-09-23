#!/bin/bash


bignum=`cat ${1:-multi_ceviche_c0.csv} | wc -l `

lownum=2

while [ 2 -lt $(( $bignum - $lownum ))  ]
do
echo LOG: now $lownum $bignum
num=$(( ( $bignum + $lownum ) / 2 ))
touch zorgluboid.csv
rm zorgluboid.csv
touch zorgluboid_plots
rm -rf zorgluboid_plots
head -n $num ${1:-multi_ceviche_c0.csv} > zorgluboid.csv

python -m nevergrad.benchmark.plotting zorgluboid.csv

if [ -f zorgluboid_plots/xpresults_all.png ]; then
echo LOG: ok at length $num
lownum=$(( ( $bignum + $lownum ) / 2 ))
else
echo LOG: fail at length $num
bignum=$(( ( $bignum + $lownum ) / 2 ))
fi
done
