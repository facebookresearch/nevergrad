#!/bin/bash

pip install img2pdf

allplots=""

# artificial noise-free single objective unconstrained or box-constrained
allplots="$allplots `ls -d *_plots/ | egrep -v 'noisy|spsa' | grep bbob | grep yabbob `"
allplots="$allplots `ls -d *_plots/ | egrep 'multimodal|deceptive'`"
allplots="$allplots `ls -d *_plots/ | egrep -v 'noisy|spsa' | grep bbob | grep -v yabbob | grep -v pen`"

# penalized
allplots="$allplots `ls -d *_plots/ | egrep -v 'noisy|spsa' | grep bbob | grep  pen`"

# tuning
allplots="$allplots `ls -d *_plots/ | egrep -v 'noisy|spsa' | grep -v bbob | egrep -v 'multimodal|deceptive' | grep -v photonics | grep -v topology | grep -v rock | grep tuning  `"

# discrete
allplots="$allplots `ls -d *_plots/ | egrep -v 'noisy|spsa' | grep -v bbob | egrep -v 'multimodal|deceptive' | grep -v photonics | grep -v topology | grep -v rock | grep -v tuning | egrep 'pbo|discr|bonn'`"

# rest of RW, besides photonics, topology, rockets
allplots="$allplots `ls -d *_plots/ | egrep -v 'noisy|spsa' | grep -v bbob | egrep -v 'multimodal|deceptive' | grep -v photonics | grep -v topology | grep -v rock | grep -v tuning | egrep -v 'pbo|discr|bonn' | grep -v multiobj | grep -v spsa ` "
# RW
allplots="$allplots `ls -d *_plots/ | egrep -v 'noisy|spsa' | grep -v bbob | grep photonics `"
allplots="$allplots `ls -d *_plots/ | egrep -v 'noisy|spsa' | grep -v bbob | grep topology`"
allplots="$allplots `ls -d *_plots/ | egrep -v 'noisy|spsa' | grep -v bbob | grep rocket `"

# multiobj 
allplots="$allplots `ls -d *_plots/ | egrep -v 'noisy|spsa' | grep -v bbob | grep multiobj`"

# Noisy optimization
allplots="$allplots `ls -d *_plots/ | egrep -i 'noisy|spsa'`"

echo $allplots

touch competition.tex bigstats.tex
rm competition.tex bigstats.tex
(
cat scripts/tex/beginning.tex
(

for v in zp_ms_bbob_plots/fight_translation_factor0.01.png_pure.png zp_ms_bbob_plots/fight_translation_factor0.1.png_pure.png zp_ms_bbob_plots/fight_translation_factor1.0.png_pure.png zp_ms_bbob_plots/fight_translation_factor10.0.png_pure.png zp_ms_bbob_plots/fight_all_pure.png zp_ms_bbob_plots/xpresults_all.png
do
 convert $v -trim +repage prout.png
 cp prout $v
 img2pdf -o ${v}.pdf $v
done
for u in $allplots
do
bestfreq=`cat ${u}/fig*.txt | grep '^[ ]*algo.*:' | head -n 1 |sed 's/(.*//g' | sed 's/.*://g'`
num=`cat ${u}/fig*.txt | grep '^[ ]*algo.*:' | wc -l `
uminus=`echo $u | sed 's/_plots.*//g'`
bestsr=`cat rnk__${uminus}_plots.cp.txt | grep '^[ ]*algo.*:' | head -n 1 |sed 's/(.*//g' | sed 's/.*://g'`
echo "\\subsubsection{`echo $u | sed 's/_plots.$//g'` (NSR:$bestsr) (Freq:$bestfreq) (num:$num)}" | sed 's/_/ /g'| sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g' | sed 's/keras/(RW)&/g' | sed 's/mltuning/(RW)&/g' | sed 's/powersystems/(RW)&/g' | sed 's/mixsimulator/(RW)&/g' | sed 's/olympus/(RW)&/g' | sed 's/double.o.seven/(RW)&/g'
timeout 10 cat scripts/txt/`echo $u | sed 's/_plots/.txt/g' | sed 's/\///g'`
(
(

convert ${u}/fight_all_pure.png -trim +repage  ${u}/fight_all_pure.pre.png
img2pdf -o ${u}/fight_all_pure.png.pdf  ${u}/fight_all_pure.pre.png
convert ${u}/xpresults_all.png -trim +repage  ${u}/xpresults_all.pre.png
img2pdf -o ${u}/xpresults_all.png.pdf ${u}/xpresults_all.pre.png
) 2>&1 | cat > logconvert${uminus}.log
) &
echo " "
echo " "
ls ${u}/*all_pure.png ${u}/xpresults_all.png | sed 's/.*/\\includegraphics[width=.99\\textwidth]{{&}}\\\\/g' 
for o in $u/fig*.txt
do
echo "\\paragraph{Ranking for normalized simple regret, `echo $uminus | sed 's/_/ /g'` }"
echo '\begin{enumerate}' ; timeout 6 cat $( echo $o | sed 's/_plots\/fight_all.png/_plots/g' | sed 's/^/rnk__/g' | grep cp.txt | sed 's/_plots.*all.png/_plots/g' ) | grep -v ranking | sed 's/[_=]/ /g' | sed 's/  algo.[0-9]*:/\\item/g' ; echo '\item[] ~\ ~' ; echo '\end{enumerate}'
echo ' '
echo ' '
echo ' '
echo ' '
echo ' '
echo ' '
echo ' '
echo ' '
echo ' '
echo ' '
echo ' '
echo "\\paragraph{Ranking for average frequency of outperforming other methods, `echo $uminus | sed 's/_/ /g'`}"
echo '\begin{enumerate}' ; timeout 6 cat $u/fig*.txt | grep -v pngranking | sed 's/[_=]/ /g' | sed 's/  algo.[0-9]*:/\\item/g' ; echo '\item[] ~\ ~' ; echo '\end{enumerate}'
done
done
) | tee competition.tex
) > dagstuhloid.tex
wait
( echo '\begin{enumerate}' ; grep -i 'subsubsection.*rw' competition.tex | sed 's/.*NSR//g' | sed 's/).*//g' | sort | uniq -c | sort -n -r  | sed 's/.*/\\item &/g' ; echo '\end{enumerate}' )> rwtex.tex

(
echo "\\section{Statistics over all benchmarks}\\label{bigstats}"
echo "We point out that NGOpt and variants are wizards (automatic algorithm selectors and combinators) created by the same authors as Nevergrad, and their (good) results might therefore be biased: we do not cheat, but we recognize that common authorship for benchmarks and algorithms imply a bias."
echo 'Of course, statistics here are a risky thing: when two codes are very close to each other, they are both penalized: we must be careful with interpretations.'

echo '\subsection{NGOpt and Base algorithms}'
echo 'Here base algorithms have no metamodel and no complex combinations. NGOpt is the only sophisticated combination: this is an analysis of NGOpt.'
for n in 1 2 3
do
echo "\\subsubsection{Number of times each algorithm was ranked among the $n first: NGOpt and base algorithms}"
echo "\\begin{itemize}"
#grep -A$n begin.enumerate dagstuhloid.tex | grep '(' | grep ')' | grep '^\\item' | sed 's/ (.*//g' | sed 's/^.item //g' | sort | uniq -c | sort -n -r | head -n 8 | sed 's/^/\\item/g'
egrep -v 'Multi|Carola|BAR|NGOpt[0-9A-Z]|NgIoh|Wiz|BIPOP|Shiwa|Meta|Micro|Tiny|SQPCMA|CMandAS2|Chain' dagstuhloid.tex  |grep -A$n begin.enumerate  | grep '(' | grep ')' | grep '^\\item' | sed 's/ (.*//g' | sed 's/^.item //g' | sort | uniq -c | sort -n -r | head -n 8 | sed 's/^/\\item/g'
echo "\\end{itemize}"
done 

echo '\subsection{Wizards, multilevels, specific standard deviations, and combinations excluded}'
echo 'The success (robustness) of quasi-opposite PSO is visible.'
for n in 1 2 3
do
echo "\\subsubsection{Number of times each algorithm was ranked among the $n first: no wizard, no combination}"
echo "\\begin{itemize}"
egrep -v 'NGOpt|Carola|BAR|Multi|BIPOP|NgIoh|Wiz|Shiwa|Meta|SQPCMA|Micro|Tiny|CMASQP|BIPOP|CMandAS2|Chain' dagstuhloid.tex  |grep -A$n begin.enumerate  | grep '(' | grep ')' | grep '^\\item' | sed 's/ (.*//g' | sed 's/^.item //g' | sort | uniq -c | sort -n -r | head -n 8 | sed 's/^/\\item/g'
echo "\\end{itemize}"
done 

echo '\subsection{Everything included}'
echo 'All strong methods are wizards, except tools based on quasi-opposite samplings.'
for n in 1 2 3
do
echo "\\subsubsection{Number of times each algorithm was ranked among the $n first: everything included}"
echo "\\begin{itemize}"
grep -A$n begin.enumerate dagstuhloid.tex | grep '(' | grep ')' | grep '^\\item' | sed 's/ (.*//g' | sed 's/^.item //g' | sort | uniq -c | sort -n -r | head -n 8 | sed 's/^/\\item/g'
echo "\\end{itemize}"
done 



) | tee bigstats.tex >> dagstuhloid.tex

listalgos=$( grep '^\\item [A-Za-z0-9]* (' dagstuhloid.tex | grep '(' | sed 's/ (.*//g' | sed 's/\\item //g' | sort | uniq )

if [[ $(find "tmp.tex.tmp" -mtime -10000 -print) ]]; then
echo skipping tmp.tex.tmp, because recent such file found.
else
touch tmp.tex.tmp
rm tmp.tex.tmp
(
echo '\section{Pairwise comparisons}'
for a in $listalgos
do
        for b in $listalgos
        do
            awins=$( egrep "\\item $a \(|\\item $b \(|\\begin{enumerate}|\\end{enumerate}" dagstuhloid.tex| egrep -A1 "\\item $a \(" | egrep "\\item $b \(" | wc -l )
            bwins=$( egrep "\\item $a \(|\\item $b \(|\\begin{enumerate}|\\end{enumerate}" dagstuhloid.tex| egrep -A1 "\\item $b \(" | egrep "\\item $a \(" | wc -l )
            total=$( echo "$awins + $bwins" | bc -l )
            if (( $total > 15 )); then
                freq=$( echo  " (100 * $awins) / ( $awins + $bwins )" | bc -l | sed 's/\..*//g' )
                if (( $freq > 60 )); then
                    echo  "$a wins vs $b with frequency  $freq per cent.\\\\"
                fi
            fi
        done
done ) > tmp.tex.tmp
fi

(
echo '\section{Conclusion}'
cat scripts/tex/conclusion.tex
cat tmp.tex.tmp
#echo '\appendix'
#echo '\section{Competence maps}'
#for u in $allplots
#do
#echo "\\subsection{`echo $u | sed 's/_plots.$//g'`}" | sed 's/_/ /g'| sed 's/aquacrop/(RW) &/g' | sed 's/rocket/(RW)&/g' | sed 's/fishing/(RW)&/g' | sed 's/MLDA/(RW)&/g' | sed 's/keras/(RW)&/g' | sed 's/mltuning/(RW)&/g' | sed 's/powersystems/(RW)&/g' | sed 's/mixsimulator/(RW)&/g' | sed 's/olympus/(RW)&/g' | sed 's/double.o.seven/(RW)&/g'
#
#for v in `grep -c none ${u}/comp*.tex | grep ':0' | sed 's/:.*//g'`
#do
#echo "\\subsubsection*{$v}" | sed 's/[_=]/ /g' | sed 's/\.tex//g'
#ls `ls $v | sed 's/\.tex/\.pdf/g'` | sed 's/.*/\\includegraphics[width=.99\\textwidth]{{&}}\\\\/g' 
#done
#done
cat scripts/tex/end.tex ) >> dagstuhloid.tex
for v in competition.tex dagstuhloid.tex
do
sed -i 's/\\subsubsection{yabbob .*}/\\subsection{Artificial noise-free single objective}&/g' $v
sed -i 's/\\subsubsection{yamegapenbbob .*}/\\subsection{Constrained BBOB variants}&/g' $v
sed -i 's/\\subsubsection{(RW)keras tuning .*}/\\subsection{Real world machine learning tuning}&/g' $v
sed -i 's/\\subsubsection{bonnans .*}/\\subsection{Discrete optimization}&/g' $v
sed -i 's/\\subsubsection{(RW) aquacrop fao .*}/\\subsection{Real world, other than machine learning}&/g' $v
sed -i 's/.*control.*//g' $v
sed -i 's/\\subsubsection{multiobjective example hd .*}/\\subsection{Multiobjective problemes}&/g' $v
sed -i 's/\\subsubsection{ranknoisy .*}/\\subsection{Noisy optimization}&/g' $v

sed -i 's/(x)/ /g' $v
done

#for u in `grep includegraphics dagstuhloid.tex | sed 's/.*{{//g' | sed 's/}}.*//g' | grep -v pdf | grep  png`
#do
#echo working on $u
#cp $u ${u}.notrim
#convert $u -trim +repage ${u}.trim.png
#cp ${u}.trim.png $u
#convert $u ${u}.pdf
#
#
#
#done
sed -i 's/.png}}/.png.pdf}}/g' dagstuhloid.tex
# ================
cp scripts/tex/biblio.bib .
pdflatex dagstuhloid.tex
bibtex dagstuhloid.aux
pdflatex dagstuhloid.tex
pdflatex dagstuhloid.tex

(
echo '<html>'
echo '<body>'
ls *.csv | sed 's/.*/<a href="&">&<\/a>/g'
echo '</body>'
echo '</html>'
) >  dagstuhloid.html


tar -zcvf texdag.tgz dagstuhloid.tex biblio.bib *plots/*all_pure.png *plots/xpresults_all.png ms_bbob_plots/fight_tran*.png *_plots/*.pdf dagstuhloid.html competition.tex bigstats.tex dagstuhloid.pdf rwtex.tex
