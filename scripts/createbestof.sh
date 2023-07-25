for u in *plots/*cp.txt
do
echo "$u" | sed 's/.*_plots/algos["&/g' | sed 's/_plots.*/"]=[/g'
head -n 7 $u | sed 's/.*://g' | sed 's/(.*//g' | sed 's/ //g' | sort | uniq | sed 's/TransitionChoice$//g' | sed 's/Choice$//g' | sed 's/Softmax$//g' | grep '[A-Za-z]'  | sed 's/.*/   "&",/g'
echo " ]"
done 
