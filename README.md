# WarehouseSim

Simple 2D Warehouse Simulation for simulating AGV and AMR perceptions of humans.
It serves as a test environment for a particle filter used to track the humans in the warehouse
by combining the perceptions of the robots.
It contains a plotter allowing for live visualization of the warehouse state and the tracker output.

## Quick start

Clone the repository.
```
$ git clone https://gitlab.lrz.de/00000000014AA74B/warehousesim.git
$ cd warehousesim
```
Create a python virtual environment and install the required packages.
Python version `3.10.12` was used for development.
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
If you want build the C++ code yourself, you will need `cmake`, `make` and the required C++ dependencies `pybind11` and `libhungarian` (`hungarian` from vcpkg).

It is recommend to use Linux and vcpkg (https://vcpkg.io/en/index.html) to help with the dependencies. Both `pybind11` (Used version `2.11.1`) and `libhungarian` (Used version `0.1.3`, `hungarian` on vcpkg) can be installed with vcpkg.
After that you only need to adapt the `CMAKE_TOOLCHAIN_FILE` in `CMakeLists.txt` to fit your local vcpkg installation.

When you have the dependencies installed, you can run `gen_chap4.sh`.
It will build the approach,
train it from scratch with newly generated training data, 
run all evaluations for the results chapter
and produce the corresponding figures.
```
$ bash gen_chap4.sh
```

## Run the simulation

The script `python/main.py` is used to run the simulation.
You can check `gen_chap4.sh` to see many examples of commands to run the simulation
with different options.

You can either set the options using a command line argument 
(`--argument value` or `--argument=value`)
or by modifying the `config.yaml` file.
If a command line argument is set, its value takes precedence over
the `config.yaml` value, but `config.yaml` remains unchanged.
So the `config.yaml` can be understood as a set of default values.

## Formatting

Format with `ruff` and `clang-format`:
```
$ ruff format && clang-format -i -style=file src/*.cpp src/*.h
```