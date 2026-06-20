cp multi_ceviche.csv multi_ceviche.`date | sed 's/ /_/g'`.csv
sed -i.tmp '/Error/d' ceviche.csv
python -m nevergrad.benchmark.plotting --max_combsize=1  multi_ceviche.csv
python -m nevergrad.benchmark.plotting --max_combsize=1  multi_ceviche_c0.csv
