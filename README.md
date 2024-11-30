# WarehouseSim

Simple 2D Warehouse Simulation for simulating AGV and AMR perceptions of humans.
It serves as a test environment for a particle filter used to track the humans in the warehouse
by combining the perceptions of the robots.
It contains a plotter allowing for live visualization of the warehouse state and the tracker output.

## Quick start

Clone the repository.
```
$ git clone https://github.com/jonasnebl/warehouseSim.git
$ cd warehouseSim
```
Create a python virtual environment and install the required packages.
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
Make sure you have the required dependencies `pybind11` and `libhungarian` (`hungarian` from vckpg) installed.
The `CMakeLists.txt` is designed to use vcpkg which is recommended.
You will have to modify the `CMAKE_TOOLCHAIN_FILE` in `CMakeLists.txt`
to fit your vcpkg installation.
If you don't use vckpg you may have to change more in `CMakeLists.txt`
so you are able to build the code.

When you have the dependencies installed,
you can create a `build` folder and run `1_setup_likelihood_matrix.sh`.
It will build the simulation and tracker and already
train the likelihood matrix for later use.
```
$ bash 1_setup_likelihood_matrix.sh
```
To train the prediction model, run `2_train_pred_model.sh`.
```
$ bash 2_train_pred_model.sh
```
Now everything is trained and you can modify `config.yaml` to 
run simulations to your liking.
```
$ python python/main.py
```
If you want to generate the evaluation plots that are also in the report,
run `3_final_eval.sh`.
```
$ bash 3_final_eval.sh
```

## Formatting

Format with `ruff` and `clang-format`:
```
$ ruff format && clang-format -i -style=file src/*.cpp src/*.h
```