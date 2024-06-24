#!/bin/bash

touch specifs.txt
rm specifs.txt
grep Specificat *.out | sed 's/.*Specifications://g' | sort | uniq  > specifs.txt

for s in `seq 0 31` 
#for s in `seq 0 31` ''
do
# echo seed${s}_
( 
echo 'from collections import defaultdict ' > plotter.py
echo -n "title = '"
echo -n `cat specifs.txt | grep "seed${s},"`
echo "'" ) >> plotter.py
echo 'import matplotlib.pyplot as plt' >> plotter.py
echo 'import numpy as np' >> plotter.py
echo 'data = defaultdict(lambda: defaultdict(list))' >> plotter.py
grep 'Algo.*seed.*__result' coevol_*.out | grep -v pkl | grep -v 'fp__' | grep `echo seed${s}_ | sed 's/seed_/./g' ` | sed 's/.*://g' | sed 's/fpll_/fplL/g' | sed 's/fipl_/fpl/g' | sed 's/fip__/fp/g' | sed 's/_/ /g' | awk '{ print "data[\"", $1, "\"][", $2, "] += [", $3, "]" }' | sed 's/budget//g' | sed 's/ loss//g' >> plotter.py

echo 'a = " AlgoRS "' >> plotter.py
echo 'x = sorted([int(d) for d in data[a].keys()])' >> plotter.py
echo 'y = [np.average([ float(d) for d in data[a][x_] ]) for x_ in x]' >> plotter.py
echo 'rs = np.min(y)' >> plotter.py
echo 'def l(x_):' >> plotter.py
echo '   return int(np.exp(int(.5 + np.log(1 + x_)/np.log(10))*np.log(10)))' >> plotter.py
echo 'allalgs=[]' >> plotter.py
echo 'for a in data.keys():' >> plotter.py
echo '   x = list(np.unique(sorted([l(d) for d in data[a].keys()])))' >> plotter.py
# echo '   x = [x_ for x_ in x if x_ < 100000]' >> plotter.py
echo '   y = []' >> plotter.py      
echo '   for r_ in x:' >> plotter.py
echo '     tmp = []' >> plotter.py
echo '     for x_ in data[a].keys():' >> plotter.py
echo '         if l(x_) == r_:'   >> plotter.py
echo '             tmp += list(data[a][x_])' >> plotter.py
#echo '     print(tmp)' >> plotter.py
echo '     y += [np.average(tmp)]' >> plotter.py
echo '   assert len(x) == len(y)' >> plotter.py
echo '   allalgs += [(float(np.min(y)), a, int(np.round(np.log(max(x))/ np.log(10))))]' >> plotter.py
#echo '   print(x)'   >> plotter.py
echo '   plt.loglog(x, y, label=a.replace("Algo", "").replace("Discrete", "Disc").replace("OnePlusOne", "1+1"))'   >> plotter.py
echo '   plt.text(x[-1], y[-1], a.replace("Lognormal", "LN").replace("Recombining", "Rec").replace("Smooth","SM").replace("Algo", "").replace("Discrete", "Disc").replace("OnePlusOne", "1+1"), rotation=30, rotation_mode="anchor")' >> plotter.py
#echo 'plt.legend()' >> plotter.py
echo 'plt.tight_layout()' >> plotter.py
echo 'plt.title(title)' >> plotter.py
echo "print('seed',  $s )" >>  plotter.py
#echo 'print(allalgs)' >> plotter.py
echo 'for idx, u in enumerate(sorted(allalgs)[:10]):' >> plotter.py
echo '    print(idx, u)' >> plotter.py
echo 'plt.savefig("coevol' $s '.png".replace(" ",""))' >> plotter.py
mv plotter.py plotter${s}.py
# echo Plotter$s
python plotter${s}.py

#ls -ctrl coevol*.png
done
