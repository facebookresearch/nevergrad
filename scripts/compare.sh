#!/bin/bash


for f in rnk*.txt
do
number=` egrep -i " ${1} | ${2} | ${4:-dsvkjnvkfdsjnvs} | ${3:-dsvnskdjvnfknsd} " $f | wc -l`
best=` egrep -i " ${1} | ${2} | ${4:-dsvkjnvkfdsjnvs} | ${3:-dsvnskdjvnfknsd} " $f | head -n 1`
if [ $number -eq "2" ]
then
   echo $f : $best
fi
done
