#!/bin/bash
pbs="`grep '^Game' ngco*.out | sed 's/.*://g' | sed 's/_.*//g' | sed 's/^Game//g' | sed 's/_.*//g' | sort | uniq | grep -iv pkl`"

for pb in $pbs
#for s in `seq 0 31` ''
do
# echo seed${s}_
( 
echo 'from collections import defaultdict ' 
echo -n "title = ' $pb ' " 
echo " "  ) > ngngplotter.py
echo 'import matplotlib.pyplot as plt' >> ngngplotter.py
echo 'import numpy as np' >> ngngplotter.py
echo 'data = defaultdict(lambda: defaultdict(list))' >> ngngplotter.py
grep 'Algo.*seed.*__result' ngcoevol_*.out | grep "Game${pb}_" | grep -v 'fp__' |  sed 's/.*://g' | sed 's/fpll_/fplL/g' | sed 's/fipl_/fpl/g' | sed 's/fip__/fp/g' | sed 's/_/ /g' | awk '{ print "data[\"", $2, "\"][", $3, "] += [", $4, "]" }' | sed 's/budget//g' | sed 's/ loss//g' >> ngngplotter.py

#echo 'a = " AlgoRS "' >> ngngplotter.py
#echo 'x = sorted([int(d) for d in data[a].keys()])' >> ngngplotter.py
#echo 'y = [np.average([ float(d) for d in data[a][x_] ]) for x_ in x]' >> ngngplotter.py
#echo 'rs = np.min(y)' >> ngngplotter.py
echo 'def l(x_):' >> ngngplotter.py
echo '   return int(np.exp(int(.5 + np.log(1 + x_)/np.log(3))*np.log(3)))' >> ngngplotter.py
echo 'allalgs=[]' >> ngngplotter.py
echo 'for all_algos in [False, True]:' >> ngngplotter.py
echo '  for a in [d for d in data.keys() if "ficpl" not in d and (all_algos or "eteLengler" in d)]:' >> ngngplotter.py
echo '   x = list(np.unique(sorted([l(d) for d in data[a].keys()])))' >> ngngplotter.py
# echo '   x = [x_ for x_ in x if x_ < 100000]' >> ngngplotter.py
echo '   y = []' >> ngngplotter.py      
echo '   for r_ in x:' >> ngngplotter.py
echo '     tmp = []' >> ngngplotter.py
echo '     for x_ in data[a].keys():' >> ngngplotter.py
echo '         if l(x_) == r_:'   >> ngngplotter.py
echo '             tmp += list(data[a][x_])' >> ngngplotter.py
#echo '     print(tmp)' >> ngngplotter.py
echo '     y += [np.average(tmp)]' >> ngngplotter.py
echo '   assert len(x) == len(y)' >> ngngplotter.py
echo '   allalgs += [(float(np.min(y)), a, int(np.round(np.log(max(x))/ np.log(10))))]' >> ngngplotter.py
#echo '   print(x)'   >> ngngplotter.py
echo '   plt.loglog(x, y, label=a.replace("Algo", "").replace("Discrete", "Disc").replace("OnePlusOne", "1+1"))'   >> ngngplotter.py
echo '   plt.text(x[-1], y[-1], a.replace("Lognormal", "LN").replace("Recombining", "Rec").replace("Smooth","SM").replace("Algo", "").replace("Discrete", "Disc").replace("OnePlusOne", "1+1"), rotation=30, rotation_mode="anchor")' >> ngngplotter.py
#echo 'plt.legend()' >> ngngplotter.py
echo '  plt.tight_layout()' >> ngngplotter.py
echo '  plt.title(title)' >> ngngplotter.py
echo "  print('game  ;" $pb "')" >>  ngngplotter.py
#echo 'print(allalgs)' >> ngngplotter.py
echo '  for idx, u in enumerate(sorted(allalgs)[:50]):' >> ngngplotter.py
echo '      print(idx, u)' >> ngngplotter.py
echo '  plt.savefig(f"{all_algos}ngcoevol' $pb '.png".replace(" ",""))' >> ngngplotter.py
mv ngngplotter.py plotter${pb}.py
# echo Plotter$s
python plotter${pb}.py

#ls -ctrl coevol*.png
done
