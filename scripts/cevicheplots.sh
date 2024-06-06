cp ceviche.csv ceviche.`date | sed 's/ /_/g'`.csv
sed -i.tmp '/Error/d' ceviche.csv
python -m nevergrad.benchmark.plotting --nomanyxp=1 ceviche.csv
