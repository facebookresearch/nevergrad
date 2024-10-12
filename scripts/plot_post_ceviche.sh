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

(
echo "import matplotlib"
echo "import matplotlib.pyplot as plt" ) > plothisto.py
for pb in 0 1 2 3 
do
(
echo "x=[]"
echo "y=[]" ) >> plothisto.py
ls -ctr pb${pb}*.png | grep -i c0c | sed 's/_/ /g' | awk '{print $5, $6}' | sed 's/fl//g' | sort -n -r | awk '{ print "x+=[", $1,"];y+=[",$2,"]" }' >> plothisto.py

(
echo "plt.loglog(x,y,'*-',label=\"pb$pb\")"
#echo "plt.plot(x,y,'*-',label=\"pb$pb\")"
) >> plothisto.py
done
(
echo "plt.legend()"
echo "plt.savefig('histo.png')"
) >>plothisto.py
python plothisto.py

