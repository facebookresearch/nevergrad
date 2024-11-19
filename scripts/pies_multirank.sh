#!/bin/bash

#python -m pip install git+https://github.com/maroofi/csvtool.git

touch pie.png 
mkdir -p poubelle
mv *pie*.png poubelle/

# Precompute ratios
for f in *.csv
do
pbname=`echo $f | sed 's/\.csv//g'`
touch log_ratio_${pbname}.log
rm log_ratio_${pbname}.log
scripts/get_average_budget_per_dim_of_csv.sh ${pbname} | awk '{print $2}' > log_ratio_${pbname}.log
done

for robust in no yes
do
(
echo 'import matplotlib'
echo 'import matplotlib.pyplot as plt'

#for k in rnk*.txt
#do
#echo -n "# $k :"
#grep ':'  $k | sed 's/.*://g' | sed 's/ (.*//g' | egrep '[a-zA-Z0-9]' | sort | uniq | wc -l
#
#done

for targetratio in . _1 0 1 2
do
echo "# Working on targetratio $targetratio"
echo "targetratio=\"`echo $targetratio | sed 's/_/\-/g'`\""
echo "targetratio_=\"`echo $targetratio | sed 's/\.//g' `\""
# Agregated or not ?
for ag in yes no
do
# How many best are kept ?
for num in 1  3
do
echo  "num=$num"
if [ "$robust" == yes ]; then
echo  "numt=$( ls */*.cp.txt | wc -l )"
else
echo  "numt=$( ls rnk* | wc -l )"
fi
echo  "labels = []"
echo  "numbers = []"
echo  "realnumbers = []"
echo  "def ref(x):"
echo  '  return x.replace("MiddlePoint","Mid").replace("Discrete", "Disc").replace("OnePlusOne", "(1+1)").replace("RandomSearch","RS").replace("Random", "Rnd").replace("OnePlusOne","(1+1)").replace("Recentering","Cent")'
echo "# $num best"

# Determinining the list of files
if [ "$robust" == yes ]; then
     echo "rob='rob'"
     list="$( ls */*.png.cp.txt )"
else
     echo "rob=''"
     list="$( ls rnk*.txt | grep -v pure )"
fi


fulllistalgos="RandomSearch AXP Cobyla PCABO SMAC3 NgIohTuned PymooBIPOP CMA PSO SQOPSO DE DiscreteLenglerOnePlusOne DiscreteOnePlusOne OnePlusOne DSproba MetaModel LognormalDiscreteOnePlusOne CauchyRandomSearch RandomScaleRandomSearchPlusMiddlePoint HullAvgMetaTuneRecentering HyperOpt NGDSRW"
listalgos="$fulllistalgos"
if [ "$ag" == yes ]; then
echo "#  pruning!"
listalgos2=""
for l in $listalgos
do   
    r=$( echo " $l " | egrep -v ' NgIohTuned | NGDSRW | AX | SMAC3 | LognormalDiscreteOnePlusOne | PCABO | PSO | PymooBIPOP | CMA | RandomSearch | RandomScaleRandomSearchPlusMiddlePoint | HullAvgMetaTuneRecentering | HyperOpt ' )
    listalgos2="$listalgos2 $r"
done
listalgos="$listalgos2"
fi

echo "# ag=$ag   listalgos=$( echo $listalgos )"


for a in $listalgos
do
#     for k in $list
#     do
#         pbname=`echo $k | sed 's/_plots.*//g' | grep -v pure | sed 's/rnk__//g' `
#         ratio=$( scripts/get_average_budget_per_dim_of_csv.sh ${pbname} | awk '{print $2}' | sed 's/\-/_/g' )
#         if [ "$ratio" == "$targetratio" ] | [ "$targetratio" == . ] ; then
#         echo "# $pbname $ratio $targetraio selected"
#         else
#         echo "# $pbname $ratio $targetraio unselected"
#         fi
#     done
echo -n "# $a is in the $num best in this number of problems:"
number=$( 
for k in $list
do  
    
    pbname=`echo $k | sed 's/_plots.*//g' | grep -v pure | sed 's/rnk__//g' |sed 's/.*\///g'`
    #echo "# Working on pbname=${pbname} by k=${k} for rob=$rob num=$num"
    #touch $( echo "Gotouch Working on pbname=${pbname} by k=${k} for rob=$rob num=$num" | sed 's/\//_/g' | sed 's/ /_/g' )
    #ratio=$( scripts/get_average_budget_per_dim_of_csv.sh ${pbname} | awk '{print $2}' | sed 's/\-/_/g' )
    ratio=`cat log_ratio_${pbname}.log | sed 's/\-/_/g'`
    if [ "$ratio" == "$targetratio" ] || [ "$targetratio" == . ] ; then
        touch sentinel__${pbname}__${ratio}_${targetratio}_yes.log
        if [ "$ag" == yes ]; then
            grep ':' $k | egrep ' RandomSearch | AXP | Cobyla | PCABO | SMAC3 | NgIohTuned | PymooBIPOP | CMA | PSO | SQOPSO | DE | DiscreteLenglerOnePlusOne | DiscreteOnePlusOne | OnePlusOne | DSproba | MetaModel | LognormalDiscreteOnePlusOne | CauchyRandomSearch | RandomScaleRandomSearchPlusMiddlePoint | HullAvgMetaTuneRecentering | HyperOpt | NGDSRW ' | grep -v ' AX ' | sed 's/.*://g' | sed 's/ (.*//g' | egrep '[a-zA-Z0-9]' | egrep -v ' NgIohTuned$| NGDSRW$| AX$| SMAC3$| LognormalDiscreteOnePlusOne$| PCABO$| PSO$| PymooBIPOP$| CMA$| RandomSearch$| RandomScaleRandomSearchPlusMiddlePoint$| HullAvgMetaTuneRecentering$| HyperOpt$' | tee check${targetratio}_${ag}_${num}_for${pbname}.log | head -n $num | grep "^ $a$"
        else
            grep ':' $k | egrep ' RandomSearch | AXP | Cobyla | PCABO | SMAC3 | NgIohTuned | PymooBIPOP | CMA | PSO | SQOPSO | DE | DiscreteLenglerOnePlusOne | DiscreteOnePlusOne | OnePlusOne | DSproba | MetaModel | LognormalDiscreteOnePlusOne | CauchyRandomSearch | RandomScaleRandomSearchPlusMiddlePoint | HullAvgMetaTuneRecentering | HyperOpt | NGDSRW ' | grep -v ' AX ' | sed 's/.*://g' | sed 's/ (.*//g' | egrep '[a-zA-Z0-9]' |tee check${targetratio}_${ag}_${num}_for${pbname}.log |  head -n $num | grep "^ $a$"
        fi
    else
        touch sentinel__${pbname}__${ratio}_${targetratio}_no.log
    fi
done | wc -l )
echo $number
if [ "$ag" == yes ]; then
echo "# AG yes $a"
echo "labels += [\" $a \" + '(' + str( $number ) + ')' ];realnumbers += [ $number ] ; numbers += [ $number + .1 ]" 
else
echo "# AG no $a"
echo "labels += [ref(\" $a \") + '(' + str( $number ) + ')' ];realnumbers += [ $number ] ; numbers += [ $number + .1 ]"
fi
done
echo 'plt.clf()'
echo 'def get_cmap(n, name="hsv"):'
echo '            return plt.cm.get_cmap(name, n)'
#echo '            return plt.cm.get_cmap(name, n)'
echo 'colors=get_cmap(len(labels))'
echo "plt.bar(labels, numbers, label=labels, color=[colors(i) for i in range(len(numbers))])"
echo 'plt.xticks(rotation=90)'
echo "title=f'How many times in the {num} best ? out of {sum(realnumbers)}'"
echo "plt.title(title)"
echo "plt.tight_layout()"
if [ "$ag" == yes ]; then
echo "plt.savefig(f'ag{targetratio_}piebar{rob}{num}.png')"
else
echo "plt.savefig(f'{targetratio_}piebar{rob}{num}.png')"
fi
echo 'plt.clf()'
echo "plt.pie(numbers, labels=labels)"
echo "title=f'How many times in the {num} best ? out of {sum(realnumbers)}'"
echo "plt.title(title)"
echo "plt.tight_layout()"
if [ "$ag" == yes ]; then
echo "plt.savefig(f'ag{targetratio_}pie{rob}{num}.png')"
else
echo "plt.savefig(f'{targetratio_}pie{rob}{num}.png')"
fi
done
done
done
) > plotpie.py

python plotpie.py
done
#sed -i 's/label.* AX .*//g' plotpie.py
#sed -i 's/label.* PSO .*//g' plotpie.py
#sed -i 's/label.* PymooBIPOP .*//g' plotpie.py
#sed -i 's/label.* CMA .*//g' plotpie.py
#sed -i 's/pie{num}/agpie{num}/g' plotpie.py
#
#python plotpie.py
