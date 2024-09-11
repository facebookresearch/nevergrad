#!/bin/bash

pbs="` cat cheat.txt | awk '{print $1}' | sort | uniq `"

for pb in $pbs
do
(
echo 'import matplotlib'
echo 'import matplotlib.pyplot as plt'
echo 'data={}'
echo 'datacheat={}'
grep $pb nocheat.txt | sort -n -k 2,2 | awk '{ print "data[",$2,"]=",$3 }'
grep $pb cheat.txt | sort -n -k 2,2 | awk '{ print "datacheat[",$2,"]=",$3 }'
echo 'x = sorted(data.keys())'
echo 'y = [data[x_] for x_ in x]'
echo 'plt.semilogx(x, y, label="' $pb '")' | sed 's/ //g'
echo 'x = sorted(datacheat.keys())'
echo 'y = [datacheat[x_] for x_ in x]'
echo 'plt.semilogx(x, y, label="' $pb '-cheat")' | sed 's/ //g'
echo 'plt.legend()'
echo 'plt.savefig("' $pb '"+".png")' | sed 's/ //g'
echo 'plt.savefig("' $pb '"+".svg")' | sed 's/ //g'
) > plotter.py
python plotter.py

done
