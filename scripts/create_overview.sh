#!/bin/bash


for u in *_plots/
do
echo $u
ls ../fullplots/${u}/fight_*.png | grep -v ',' | grep -v fight_all | grep pure | sed 's/.*fight_//g' | sed 's/.png_pure.png//g'
done
