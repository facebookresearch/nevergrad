#!/bin/bash


for v in *_plots/
do
u=`echo $v | sed 's/_plots\///g'`
echo "================= ${u}"

if [ ! -f scripts/txt/${u}.txt ]
then
echo Creating scripts/txt/${u}.txt
(
echo $u | sed 's/_plots.*//g'
ls ${v}/fight_*.png | grep -v ',' | grep -v fight_all | grep pure | sed 's/.*fight_//g' | sed 's/.png_pure.png//g'
) > scripts/txt/${u}.txt
    else
        ls -ctrl scripts/txt/${u}.txt
        fi
done
