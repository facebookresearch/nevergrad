#!/bin/bash 

echo 'If an argument is given, this script launches the corresponding experiment.'
echo 'Example: ./autorun.sh powersystems'
echo 'By default, this script launches all experiments in experiments, frozenexperiments, and a few ones in Dfoptim.'
echo 'Results are exported to https://dl.fbaipublicfiles.com/nevergrad/allxps/list.html'

./autorun_compute.sh $*

sleep 3600
./autorun_plot.sh
