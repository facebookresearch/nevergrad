#!/bin/bash
(
cat scripts/tex/beginning.tex
for u in *plots/
do
echo "\\subsection{`echo $u | sed 's/_plots.$//g'`}" | sed 's/_/ /g' | sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g' | sed 's/keras/(RW)&/g' | sed 's/mltuning/(RW)&/g'
ls ${u}/*all.png | sed 's/.*/\\includegraphics[width=.8\\textwidth]{{&}}\\\\/g'
echo "\\subsection{`echo $u | sed 's/_plots.$//g'`}" | sed 's/_/ /g'| sed 's/powersystems/(RW)&/g' | sed 's/mixsimulator/(RW)&/g' | sed 's/olympus/(RW)&/g'
ls ${u}/*all.png | sed 's/.*/\\includegraphics[width=.8\\textwidth]{{&}}\\\\/g' 
done
echo '\section{Conclusion}'
cat scripts/tex/conclusion.tex
echo '\appendix'
echo '\section{Competence maps}'
for u in *plots/
do
echo "\\subsection{`echo $u | sed 's/_plots.$//g'`}" | sed 's/_/ /g' | sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g' | sed 's/keras/(RW)&/g' | sed 's/mltuning/(RW)&/g'| sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g' | sed 's/keras/(RW)&/g' | sed 's/mltuning/(RW)&/g' 
ls ${u}/comp*.pdf | sed 's/.*/\\includegraphics[width=.8\\textwidth]{{&}}\\\\/g' 
done
cat scripts/tex/end.tex ) > dagstuhloid.tex
cp scripts/tex/biblio.bib .
pdflatex dagstuhloid.tex
bibtex dagstuhloid.aux
pdflatex dagstuhloid.tex
