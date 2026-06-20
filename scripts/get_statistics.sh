#!/bin/bash

pbs=`ls rnk*.txt | sed 's/rnk__//g' | sed 's/_plots.*//g'`

(
echo '\begin{tabular}{|p{2cm}|c|c|c|}'
echo '\hline'
echo 'Best algorithm & Problem name & Dimension & Budget \\'
echo '\hline'

for pb in $pbs
do

dim=$( ls -ctr ${pb}_plots/fig*dim*.png | grep -v ',' | grep pure | grep -v block | grep -v usefu | sed 's/.*dimension//g' | sed 's/.png.*//g' | awk '{ sum += $1 } END {print sum/NR}' )

budget=$( ls -ctr ${pb}_plots/fig*budg*.png | grep -v ',' | sed 's/.*budget//g' | grep pure | sed 's/.png.*//g' | awk '{ sum += $1 } END { print(sum/NR) }' )
echo GO $pb $dim $budget $( grep algo rnk__${pb}_plots.cp.txt | grep ':' | head -n 1 | sed 's/.*: //g' | sed 's/ (.*//g' ) 
done 2>&1 | grep '^GO' | sed 's/^GO //g' | grep ' .* .* ' | sort -k 4,4 | awk ' { if ( $4 != last ) { print "\\hline " ; print $4, "&", $1, "&", $2, "&", $3, "\\\\"; last=$4 } else {print  "&", $1, "&", $2, "&", $3, "\\\\" }  }' | sed 's/_/ /g'
echo '\hline'
echo '\end{tabular}'
) > agtable.tex
