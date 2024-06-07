cp multi_ceviche.csv multi_ceviche.`date | sed 's/ /_/g'`.csv
sed -i.tmp '/Error/d' ceviche.csv
python -m nevergrad.benchmark.plotting --nomanyxp=1 multi_ceviche.csv
