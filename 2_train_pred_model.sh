# generating training data and train models with both magic and non-magic data
python python/main.py --T_simulation=172800 --N_particles=100 --folder="4humans_4AMRs_48h_100part" --record_training_data=true
python python/calc_pred_model_params.py --folders "4humans_4AMRs_48h_100part" --use_magic=true
mv "models/successor_edge_probabilities.json" "models/successor_edge_probabilities_magic.json"
mv "models/duration_params.json" "models/duration_params_magic.json"
python python/calc_pred_model_params.py --folders "4humans_4AMRs_48h_100part" --use_magic=false

# build again with trained model
python python/prepare_warehouse_data.py
cd build && make && cd ..