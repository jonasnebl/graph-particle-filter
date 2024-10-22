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
Generate a Header-File containing the topological information about the warehouse
as well as model parameters for the ParticleTracker's prediction model.
```
$ python python/prepare_warehouse_data.py
```
Make a build folder and build the project using CMake.
Make sure you have the required dependencies `pybind11` and `hungarian` (from vckpg) installed.
Vcpkg is recommended for dependency management.
```
$ mkdir build && cd build && cmake .. && make && cd ..
```
Now you can run the simulation. Use `config.yaml` to set simulation parameters.
```
$ python python/main.py
```

## Formatting

Format with `ruff` and `clang-format`:
```
$ ruff format && clang-format -i -style=file src/*.cpp src/*.h
```