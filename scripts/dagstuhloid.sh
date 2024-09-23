#!/bin/bash
#SBATCH --job-name=dagstuhloid
#SBATCH --output=dagstuhloid_%A_%a.out
#SBATCH --error=dagstuhloid_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH -a 0-73


tasks=(nozp_noms_bbob multi_structured_ng_full_gym lsgo smallbudget_lsgo ng_gym sequential_fastgames small_deterministic_gym_multi tiny_deterministic_gym_multi zp_pbbob zp_ms_bbob aquacrop_fao bonnans deceptive double_o_seven fishing instrum_discrete keras_tuning mldakmeans mltuning mono_rocket multimodal multiobjective_example multiobjective_example_hd multiobjective_example_many_hd naive_seq_keras_tuning naive_seq_mltuning naive_veryseq_keras_tuning naivemltuning nano_naive_seq_mltuning nano_naive_veryseq_mltuning nano_seq_mltuning nano_veryseq_mltuning neuro_oneshot_mltuning pbbob pbo_reduced_suite reduced_yahdlbbbob rocket seq_keras_tuning seq_mltuning sequential_instrum_discrete sequential_topology_optimization spsa_benchmark topology_optimization ultrasmall_photonics ultrasmall_photonics2 veryseq_keras_tuning yabbob yabigbbob yaboundedbbob yaboxbbob yahdbbob yahdnoisybbob yamegapenbbob yamegapenboundedbbob yamegapenboxbbob yanoisybbob yaonepenbbob yaonepenboundedbbob yaonepenboxbbob yaonepennoisybbob yaonepenparabbob yaonepensmallbbob yaparabbob yapenbbob yapenboundedbbob yapenboxbbob yapennoisybbob yapenparabbob yapensmallbbob yasmallbbob yatinybbob yatuningbbob ms_bbob ranknoisy powersystems verysmall_photonics verysmall_photonics2)
#tasks=(ng_gym small_deterministic_gym_multi tiny_deterministic_gym_multi zp_pbbob zp_ms_bbob aquacrop_fao bonnans deceptive double_o_seven fishing instrum_discrete keras_tuning mldakmeans mltuning mono_rocket multimodal multiobjective_example multiobjective_example_hd multiobjective_example_many_hd naive_seq_keras_tuning naive_seq_mltuning naive_veryseq_keras_tuning naivemltuning nano_naive_seq_mltuning nano_naive_veryseq_mltuning nano_seq_mltuning nano_veryseq_mltuning neuro_oneshot_mltuning pbbob pbo_reduced_suite reduced_yahdlbbbob rocket seq_keras_tuning seq_mltuning sequential_instrum_discrete sequential_topology_optimization spsa_benchmark topology_optimization ultrasmall_photonics ultrasmall_photonics2 veryseq_keras_tuning yabbob yabigbbob yaboundedbbob yaboxbbob yahdbbob yahdnoisybbob yamegapenbbob yamegapenboundedbbob yamegapenboxbbob yanoisybbob yaonepenbbob yaonepenboundedbbob yaonepenboxbbob yaonepennoisybbob yaonepenparabbob yaonepensmallbbob yaparabbob yapenbbob yapenboundedbbob yapenboxbbob yapennoisybbob yapenparabbob yapensmallbbob yasmallbbob yatinybbob yatuningbbob ms_bbob ranknoisy powersystems verysmall_photonics verysmall_photonics2)
task=${tasks[SLURM_ARRAY_TASK_ID]}

echo task attribution $SLURM_ARRAY_TASK_ID $task
echo Keras/TF versions:
pip show keras tensorflow tensorflow-estimator

conda info

echo Starting at
date
# num_workers is the number of processes. Maybe use a bit more than the number of cores at the line "cpus-per-task"
# above.
python -m nevergrad.benchmark $task --num_workers=73 2>&1 | tail -n 50
python -m nevergrad.benchmark $task --num_workers=73 2>&1 | tail -n 50
python -m nevergrad.benchmark $task --num_workers=73 2>&1 | tail -n 50
python -m nevergrad.benchmark $task --num_workers=73 2>&1 | tail -n 50
python -m nevergrad.benchmark $task --num_workers=73 2>&1 | tail -n 50
python -m nevergrad.benchmark $task --num_workers=73 2>&1 | tail -n 50
python -m nevergrad.benchmark $task --num_workers=73 2>&1 | tail -n 50
python -m nevergrad.benchmark $task --num_workers=73 2>&1 | tail -n 50
echo task over $SLURM_ARRAY_TASK_ID $task
echo Finishing at
date
