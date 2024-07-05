#!/bin/bash

python -m nevergrad.benchmark.plotting multi_ceviche_c0.csv --max_combsize=1
cat multi_ceviche_c0.csv | sed 's/.*,//g' | sort | uniq -c | sort -n ; wc -l multi_ceviche_c0.csv 

