# final evaluation
python python/main.py --T_simulation=3600 --N_particles=10000 --folder="4humans_4AMRs_1h_10000part"

# generate figures to evaluate everything
python python/figures.py \
    --N_humans_folder "4humans_4AMRs_3h_1part" \
    --training_folder "4humans_4AMRs_48h_100part" \
    --results_folder "4humans_4AMRs_1h_10000part"