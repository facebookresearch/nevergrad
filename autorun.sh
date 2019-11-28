#!/bin/bash 

listxp=`grep -i1 "^def" nevergrad/benchmark/*experiments.py | grep -i1 '@regis' | grep ':def' | sed 's/.*:def //g' | sed 's/(.*//g'`
for xp in $listxp
do
	echo "Experiment $xp ==============================="
done
