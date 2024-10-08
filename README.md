# WarehouseSim

Simple 2D Warehouse Simulation for simulating AGV and AMR perceptions of humans.
Test environment for a particle filter used to track the humans in the warehouse
by combining the perceptions of the robots.

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
Generate a Header-File containing the information about the warehouse.
```
$ python python/prepare_warehouse_data.py
```
Make a build folder and build the project using CMake.
Vcpkg is recommended for dependency management,
install `pybind11`, `nlohmann_json` and `hungarian`.
```
$ mkdir build && cd build && cmake .. && make && cd ..
```
Now you can run the simulation.
```
$ python python/main.py
```

## Formatting

Format with `ruff` and `clang-format`:
```
$ ruff format && clang-format -i -style=file src/*.cpp
```