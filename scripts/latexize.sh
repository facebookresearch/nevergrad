#!/bin/bash
(
cat scripts/tex/beginning.tex
for u in *plots/
do
echo "\\subsection{`echo $u | sed 's/_plots.$//g'`}" | sed 's/_/ /g'
ls ${u}/*all.png | sed 's/.*/\\includegraphics[width=.8\\textwidth]{{&}}\\\\/g' | sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g'
done
echo '\section{Conclusion}'
cat scripts/tex/conclusion.tex
echo '\appendix'
echo '\section{Competence maps}'
for u in *plots/
do
echo "\\subsection{`echo $u | sed 's/_plots.$//g'`}" | sed 's/_/ /g'
ls ${u}/comp*.pdf | sed 's/.*/\\includegraphics[width=.8\\textwidth]{{&}}\\\\/g' | sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g'
done
cat scripts/tex/end.tex ) > dagstuhloid.tex

pdflatex dagstuhloid.tex
