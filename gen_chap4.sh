#!/bin/bash

# This script run's everything to generate the chapter 4 results metrics and figures
# It start's fully from scratch including the training of the likelihood matrix
# and the training of the prediction model
# Approximate runtime: 2 hours

# clear old trained models and logs 
# WILL DELETE ALL MODELS AND LOGS: IF IN DOUBT, COMMENT OUT
rm -r logs && mkdir logs
rm -r models && mkdir models

# generate initial model params without training data
python python/calc_pred_model_params.py --train_likelihood_matrix=true
python python/prepare_warehouse_data.py

# fresh build to avoid "undefined symbol: _Py_RefTotal" error
rm -r build && mkdir build && cd build && cmake .. && make && cd ..

# train likelihood matrix
python python/main.py --N_humans=1 --folder="1humans_4AMRs_8h_1part_likemat" --allow_warehouse_leaving=false --T_simulation=28800 --N_particles=1 
python python/main.py --N_humans=2 --folder="2humans_4AMRs_8h_1part_likemat" --allow_warehouse_leaving=false --T_simulation=28800 --N_particles=1 
python python/main.py --N_humans=3 --folder="3humans_4AMRs_8h_1part_likemat" --allow_warehouse_leaving=false --T_simulation=28800 --N_particles=1 
python python/main.py --N_humans=4 --folder="4humans_4AMRs_8h_1part_likemat" --allow_warehouse_leaving=false --T_simulation=28800 --N_particles=1 
python python/main.py --N_humans=5 --folder="5humans_4AMRs_8h_1part_likemat" --allow_warehouse_leaving=false --T_simulation=28800 --N_particles=1 
python python/main.py --N_humans=6 --folder="6humans_4AMRs_8h_1part_likemat" --allow_warehouse_leaving=false --T_simulation=28800 --N_particles=1 
python python/main.py --N_humans=7 --folder="7humans_4AMRs_8h_1part_likemat" --allow_warehouse_leaving=false --T_simulation=28800 --N_particles=1 

python python/calc_pred_model_params.py --train_likelihood_matrix=true --likelihood_matrix_folders \
    "1humans_4AMRs_8h_1part_likemat" \
    "2humans_4AMRs_8h_1part_likemat" \
    "3humans_4AMRs_8h_1part_likemat" \
    "4humans_4AMRs_8h_1part_likemat" \
    "5humans_4AMRs_8h_1part_likemat" \
    "6humans_4AMRs_8h_1part_likemat" \
    "7humans_4AMRs_8h_1part_likemat"

# build again with new likelihood matrix
python python/prepare_warehouse_data.py
cd build && make && cd ..

# evaluation of number of humans in the warehouse estimate
_5minfolder="4humans_4AMRs_3h_1part_5minwindow"
_10minfolder="4humans_4AMRs_3h_1part_10minwindow"

python python/main.py --window_length=300 --folder="$_5minfolder" --T_simulation=10800 --N_particles=1
mkdir -p logs/"$_10minfolder" && cp -a logs/"$_5minfolder"/. logs/"$_10minfolder"
python python/main.py --run_new_simulation=false --window_length=600 --folder="$_10minfolder" --T_simulation=10800 --N_particles=1

# generating training data and train models with both magic and non-magic data
python python/main.py --T_simulation=172800 --N_particles=100 --folder="4humans_4AMRs_48h_100part" --record_training_data=true
python python/calc_pred_model_params.py --folders "4humans_4AMRs_48h_100part" --use_magic=true
mv "models/successor_edge_probabilities.json" "models/successor_edge_probabilities_magic.json"
mv "models/duration_params.json" "models/duration_params_magic.json"
python python/calc_pred_model_params.py --folders "4humans_4AMRs_48h_100part" --use_magic=false

# build again with trained model
python python/prepare_warehouse_data.py
cd build && make && cd ..

# final evaluation
python python/main.py --T_simulation=7200 --N_particles=10000 --folder="4humans_4AMRs_2h_10000part"

# generate figures to evaluate everything
python python/figures.py \
    --N_humans_folder_short "4humans_4AMRs_3h_1part_5minwindow" \
    --N_humans_folder_long "4humans_4AMRs_3h_1part_10minwindow" \
    --training_folder "4humans_4AMRs_48h_100part" \
    --results_folder "4humans_4AMRs_2h_10000part"

# copy figures to windows folder for the report
bash figures/copy_to_windows.she