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
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
Make sure you have the required C++ dependencies `pybind11` and `libhungarian` (`hungarian` from vcpkg) installed.
The `CMakeLists.txt` is designed to use vcpkg on Linux which is recommended.
You will have to modify the `CMAKE_TOOLCHAIN_FILE` setting in `CMakeLists.txt`
to fit your vcpkg installation.
If you don't use vckpg you may have to change more in `CMakeLists.txt` to build the program.

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