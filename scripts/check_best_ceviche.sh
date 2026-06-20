#!/bin/bash


for k in 0 1 2 3 
do
echo " ============================ $k ========================== "
echo -n $k basic:
ls -ctr pb*.npy | grep -v fie | grep budg | sed 's/_/ /g' | sort -n -r -k 4,4 | sed 's/ /_/g' | grep pb$k | tail -n 1
echo -n $k wstar:
ls -ctr WSpb*.npy |grep -v fie |  grep budg | sed 's/_/ /g' | sort -n -r -k 4,4 | sed 's/ /_/g' | grep pb$k | tail -n 1
echo -n $k both :
ls -ctr pb*.npy WSpb*.npy |grep -v fie |  grep budg | sed 's/_/ /g' | sort -n -r -k 4,4 | sed 's/ /_/g' | grep pb$k | tail -n 1

done

