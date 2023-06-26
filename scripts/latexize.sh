#!/bin/bash

allplots=""
allplots="$allplots `ls -d *_plots/ | grep bbob | grep yabbob `"
allplots="$allplots `ls -d *_plots/ | grep bbob | grep -v yabbob | grep -v pen`"
allplots="$allplots `ls -d *_plots/ | grep bbob | grep  pen`"
allplots="$allplots `ls -d *_plots/ | grep -v bbob | grep tuning`"
allplots="$allplots `ls -d *_plots/ | grep -v bbob | grep -v tuning | egrep 'pbo|discr|bonn'`"
allplots="$allplots `ls -d *_plots/ | grep -v bbob | grep -v tuning | egrep -v 'pbo|discr|bonn'`"
echo $allplots

(
cat scripts/tex/beginning.tex
for u in $allplots
do
echo "\\subsection{`echo $u | sed 's/_plots.$//g'`}" | sed 's/_/ /g'| sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g' | sed 's/keras/(RW)&/g' | sed 's/mltuning/(RW)&/g' | sed 's/powersystems/(RW)&/g' | sed 's/mixsimulator/(RW)&/g' | sed 's/olympus/(RW)&/g'
ls ${u}/*all.png | sed 's/.*/\\includegraphics[width=.8\\textwidth]{{&}}\\\\/g' 
done
echo '\section{Conclusion}'
cat scripts/tex/conclusion.tex
echo '\appendix'
echo '\section{Competence maps}'
for u in $allplots
do
echo "\\subsection{`echo $u | sed 's/_plots.$//g'`}" | sed 's/_/ /g'| sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g' | sed 's/keras/(RW)&/g' | sed 's/mltuning/(RW)&/g' | sed 's/powersystems/(RW)&/g' | sed 's/mixsimulator/(RW)&/g' | sed 's/olympus/(RW)&/g'

for v in `grep -c none ${u}/comp*.tex | grep ':0' | sed 's/:.*//g'`
do
echo "\\subsubsection{$v}" | sed 's/[_=]/ /g' | sed 's/\.tex//g'
ls `ls $v | sed 's/\.tex/\.pdf/g'` | sed 's/.*/\\includegraphics[width=.8\\textwidth]{{&}}\\\\/g' 
done
done
cat scripts/tex/end.tex ) > dagstuhloid.tex
cp scripts/tex/biblio.bib .
pdflatex dagstuhloid.tex
bibtex dagstuhloid.aux
pdflatex dagstuhloid.tex
pdflatex dagstuhloid.tex
