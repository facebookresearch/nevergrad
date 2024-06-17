#!/bin/bash


for s in `seq 0 31` ''
do
echo seed${s}_
echo 'from collections import defaultdict ' > plotter.py
echo 'import matplotlib.pyplot as plt' >> plotter.py
echo 'import numpy as np' >> plotter.py
echo 'data = defaultdict(lambda: defaultdict(list))' >> plotter.py
grep 'Algo.*seed.*__result' coevol_*.out | grep -v 'fp__' | grep `echo seed${s}_ | sed 's/seed_/./g' ` | sed 's/.*://g' | sed 's/fip__/fp/g' | sed 's/_/ /g' | awk '{ print "data[\"", $1, "\"][", $2, "] += [", $3, "]" }' | sed 's/budget//g' | sed 's/ loss//g' >> plotter.py

echo 'a = " AlgoRS "' >> plotter.py
echo 'x = sorted([int(d) for d in data[a].keys()])' >> plotter.py
echo 'y = [np.average([ float(d) for d in data[a][x_] ]) for x_ in x]' >> plotter.py
echo 'rs = np.min(y)' >> plotter.py
echo 'for a in data.keys():' >> plotter.py
echo '   x = sorted([int(d) for d in data[a].keys()])' >> plotter.py
echo '   print(x)'   >> plotter.py
echo '   y = [np.average([ float(d)/rs for d in data[a][x_] ]) for x_ in x]'   >> plotter.py
echo '   x = [int(np.exp(int(np.log(10 + x_)/np.log(10))*np.log(10))) for x_ in x]' >> plotter.py
echo '   plt.loglog(x, y, label=a)'   >> plotter.py
echo '   plt.text(x[-1], y[-1], a, rotation=30, rotation_mode="anchor")' >> plotter.py
#echo 'plt.legend()' >> plotter.py
echo 'plt.tight_layout()' >> plotter.py
echo 'plt.savefig("coevol' $s '.png".replace(" ",""))' >> plotter.py
mv plotter.py plotter${s}.py
echo Plotter$s
python plotter${s}.py

ls -ctrl coevol*.png
done
