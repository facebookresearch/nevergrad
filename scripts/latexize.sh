#!/bin/bash

allplots=""

# artificial noise-free single objective unconstrained or box-constrained
allplots="$allplots `ls -d *_plots/ | grep -v 'noisy|spsa' | grep bbob | grep yabbob `"
allplots="$allplots `ls -d *_plots/ | egrep 'multimodal|deceptive'`"
allplots="$allplots `ls -d *_plots/ | grep -v 'noisy|spsa' | grep bbob | grep -v yabbob | grep -v pen`"

# penalized
allplots="$allplots `ls -d *_plots/ | grep -v 'noisy|spsa' | grep bbob | grep  pen`"

# tuning
allplots="$allplots `ls -d *_plots/ | grep -v 'noisy|spsa' | grep -v bbob | egrep -v 'multimodal|deceptive' | grep -v photonics | grep -v topology | grep -v rock | grep tuning  `"

# discrete
allplots="$allplots `ls -d *_plots/ | grep -v 'noisy|spsa' | grep -v bbob | egrep -v 'multimodal|deceptive' | grep -v photonics | grep -v topology | grep -v rock | grep -v tuning | egrep 'pbo|discr|bonn'`"

# rest of RW, besides photonics, topology, rockets
allplots="$allplots `ls -d *_plots/ | grep -v 'noisy|spsa' | grep -v bbob | egrep -v 'multimodal|deceptive' | grep -v photonics | grep -v topology | grep -v rock | grep -v tuning | egrep -v 'pbo|discr|bonn' | grep -v multiobj | grep -v spsa ` "
# RW
allplots="$allplots `ls -d *_plots/ | grep -v 'noisy|spsa' | grep -v bbob | grep photonics `"
allplots="$allplots `ls -d *_plots/ | grep -v 'noisy|spsa' | grep -v bbob | grep topology`"
allplots="$allplots `ls -d *_plots/ | grep -v 'noisy|spsa' | grep -v bbob | grep rocket `"

# multiobj 
allplots="$allplots `ls -d *_plots/ | grep -v 'noisy|spsa' | grep -v bbob | grep multiobj`"

# Noisy optimization
allplots="$allplots `ls -d *_plots/ | egrep -i 'noisy|spsa'`"

echo $allplots

(
cat scripts/tex/beginning.tex
for u in $allplots
do
echo "\\subsubsection{`echo $u | sed 's/_plots.$//g'`}" | sed 's/_/ /g'| sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g' | sed 's/keras/(RW)&/g' | sed 's/mltuning/(RW)&/g' | sed 's/powersystems/(RW)&/g' | sed 's/mixsimulator/(RW)&/g' | sed 's/olympus/(RW)&/g' | sed 's/double.o.seven/(RW)&/g'
cat scripts/txt/`echo $u | sed 's/_plots/.txt/g'`
echo '\begin{enumerate}' ; cat $u/fig*.txt | grep -v pngranking | sed 's/[_=]/ /g' | sed 's/  algo.[0-9]*:/\\item/g' ; echo '\item[] ~\ ~' ; echo '\end{enumerate}'
ls ${u}/*all_pure.png ${u}/xpresults_all.png | sed 's/.*/\\includegraphics[width=.8\\textwidth]{{&}}\\\\/g' 
done
echo '\section{Conclusion}'
cat scripts/tex/conclusion.tex
echo '\appendix'
echo '\section{Competence maps}'
for u in $allplots
do
echo "\\subsection{`echo $u | sed 's/_plots.$//g'`}" | sed 's/_/ /g'| sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g' | sed 's/keras/(RW)&/g' | sed 's/mltuning/(RW)&/g' | sed 's/powersystems/(RW)&/g' | sed 's/mixsimulator/(RW)&/g' | sed 's/olympus/(RW)&/g' | sed 's/double.o.seven/(RW)&/g'

for v in `grep -c none ${u}/comp*.tex | grep ':0' | sed 's/:.*//g'`
do
echo "\\subsubsection*{$v}" | sed 's/[_=]/ /g' | sed 's/\.tex//g'
ls `ls $v | sed 's/\.tex/\.pdf/g'` | sed 's/.*/\\includegraphics[width=.8\\textwidth]{{&}}\\\\/g' 
done
done
cat scripts/tex/end.tex ) > dagstuhloid.tex
sed -i 's/\\subsubsection{yabbob}/\\subsection{Artificial noise-free single objective}&/g' dagstuhloid.tex
sed -i 's/\\subsubsection{yamegapenbbob}/\\subsection{Constrained BBOB variants}&/g' dagstuhloid.tex
sed -i 's/\\subsubsection{(RW)keras tuning}/\\subsection{Real world machine learning tuning}&/g' dagstuhloid.tex
sed -i 's/\\subsubsection{bonnans}/\\subsection{Discrete optimization}&/g' dagstuhloid.tex
sed -i 's/\\subsubsection{(RW) aquacrop fao}/\\subsection{Real world, other than machine learning}&/g' dagstuhloid.tex
sed -i 's/.*control.*//g' dagstuhloid.tex
sed -i 's/\\subsubsection{multiobjective example hd}/\\subsection{Multiobjective problemes}&/g' dagstuhloid.tex
sed -i 's/\\subsubsection{spsa benchmark}/\\subsection{Noisy optimization}&/g' dagstuhloid.tex
cp scripts/tex/biblio.bib .
pdflatex dagstuhloid.tex
bibtex dagstuhloid.aux
pdflatex dagstuhloid.tex
pdflatex dagstuhloid.tex
