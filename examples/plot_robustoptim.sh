#!/bin/bash
pbs="`grep '^Game' ngrob*.out | sed 's/.*://g' | sed 's/_.*//g' | sed 's/^Game//g' | sed 's/_.*//g' | sort | uniq | grep -iv pkl`"

for pb in $pbs
#for s in `seq 0 31` ''
do
echo plotting $pb
# echo seed${s}_
( 
echo 'from collections import defaultdict ' 
echo -n "title = ' $pb ' " 
echo " "  ) > plotter.py
echo 'import matplotlib.pyplot as plt' >> plotter.py
echo 'import numpy as np' >> plotter.py
echo 'for ag in [False, True]:' >> plotter.py
echo ' data = defaultdict(lambda: defaultdict(list))' >> plotter.py
echo ' data2 = defaultdict(lambda: defaultdict(list))' >> plotter.py
grep 'Algo.*seed.*__result' ngrob*.out | grep -v Chain | grep "Game${pb}_" | grep -v 'fp__' |  sed 's/.*://g' | sed 's/fpll_/fplL/g' | sed 's/fipl_/fpl/g' | sed 's/fip__/fp/g' | sed 's/_/ /g' | awk '{ print " data[\"", $2, "\"][", $3, "] += [", $4, "]" }' | sed 's/budget//g' | sed 's/ loss//g' >> plotter.py

# a = np.random.choice([0, 10, 20, 50, 75, 100, 200])
(
echo ' if ag:' 
echo '   for k3 in data:'
echo '    k = k3.replace(" ", "")'
echo '    if k[-3:] == "200":'
echo '        k2 = 200'
echo '    elif k[-3:] == "100":'
echo '        k2 = 100'
echo '    elif k[-2:] == "75":'
echo '        k2 = 75'
echo '    elif k[-2:] == "75":'
echo '        k2 = 75'
echo '    elif k[-2:] == "50":'
echo '        k2 = 50'
echo '    elif k[-2:] == "20":'
echo '        k2 = 20'
echo '    elif k[-2:] == "10":'
echo '        k2 = 10'
echo '    elif k[-1:] == "0":'
echo '        k2 = 0'
echo '    else:'
echo '        assert False, k3'
echo '    k2 = str(k2)'
echo '    for l in data[k3]:'
echo '     data2[k2][l] += data[k3][l]'
echo '   data = data2'
 ) >> plotter.py
echo ' def l(x_):' >> plotter.py
echo '    return int(np.exp(int(.5 + np.log(1 + x_)/np.log(10))*np.log(10)))' >> plotter.py
echo ' allalgs=[]' >> plotter.py
echo ' plt.clf()' >> plotter.py
echo ' bestscores = []' >> plotter.py
echo ' for a in data.keys():' >> plotter.py
echo '    x = list(np.unique(sorted([l(d) for d in data[a].keys()])))' >> plotter.py
# echo '   x = [x_ for x_ in x if x_ < 100000]' >> plotter.py
echo '    y = []' >> plotter.py      
echo '    z = []' >> plotter.py      
echo '    for r_ in x:' >> plotter.py
echo '      tmp = []' >> plotter.py
echo '      for x_ in data[a].keys():' >> plotter.py
echo '          if l(x_) == r_:'   >> plotter.py
echo '              tmp += list(data[a][x_])' >> plotter.py
#echo '     print(tmp)' >> plotter.py
echo '      y += [np.median(tmp)]' >> plotter.py
echo '      z += [len(tmp)]' >> plotter.py
echo '    assert len(x) == len(y)' >> plotter.py
echo '    allalgs += [(float(np.min(y)), a, int(np.round(np.log(max(x))/ np.log(10))))]' >> plotter.py
#echo '   print(x)'   >> plotter.py
echo '    idx = [i for i in range(len(x)) if z[i] >= 2]' >> plotter.py
echo '    x = [x[i] for i in idx]' >> plotter.py
echo '    y = [y[i] for i in idx]' >> plotter.py
echo '    if len(x) > 1:' >> plotter.py
echo '     bestscores += [(min(y), a)]' >> plotter.py
echo '    if (not ag) and ("AlmostRotationInvariantDEAndBigPop" not in a) and ("SA" not in a) and ("ticNois" not in a):' >> plotter.py
#echo '        print("Skipping", a)' >> plotter.py
echo '        continue' >> plotter.py
echo '    if len(x) > 1:' >> plotter.py
echo '     plt.loglog(x, y, label=a.replace("Algo", "").replace("Discrete", "Disc").replace("OnePlusOne", "1+1"))'   >> plotter.py
echo '     plt.text(x[-1], y[-1], a.replace("Lognormal", "LN").replace("Recombining", "Rec").replace("Smooth","SM").replace("Algo", "").replace("Discrete", "Disc").replace("OnePlusOne", "1+1"), rotation=30, rotation_mode="anchor")' >> plotter.py
echo ' if not ag:' >> plotter.py
echo '       print("LISTBEST:")' >> plotter.py
echo '       print(sorted(bestscores)[:10])' >> plotter.py
#echo 'plt.legend()' >> plotter.py
echo ' plt.tight_layout()' >> plotter.py
echo ' plt.title(title)' >> plotter.py
echo " print('game  ;" $pb "')" >>  plotter.py
#echo 'print(allalgs)' >> plotter.py
echo ' for idx, u in enumerate(sorted(allalgs)[:50]):' >> plotter.py
echo '     print(idx, u)' >> plotter.py
echo ' plt.savefig(f"ngrobust{str(ag)}' $pb '.png".replace(" ",""))' >> plotter.py
mv plotter.py plotter${pb}.py
# echo Plotter$s
python plotter${pb}.py > results_${pb}.log

#ls -ctrl coevol*.png
done
