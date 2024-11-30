# clear old trained models
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
python python/main.py --T_simulation=10800 --N_particles=1 --folder="4humans_4AMRs_3h_1part"