#!/bin/bash

N=10
for i in `seq 0 $(( $N - 1 ))`
do
#python -u instrumentations_examples.py $i 20
sbatch --partition=learnlab --time=72:00:00 --job-name=tt${i}inst$N --gres=gpu:0 --cpus-per-task=50 --wrap="python -u instrumentations_examples.py $i $N"
done
