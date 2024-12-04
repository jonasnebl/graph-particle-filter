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
Make sure you have the required C++ dependencies `pybind11` and `libhungarian` (`hungarian` from vckpg) installed.
The `CMakeLists.txt` is designed to use vcpkg which is recommended.
You will have to modify the `CMAKE_TOOLCHAIN_FILE` in `CMakeLists.txt`
to fit your vcpkg installation.
If you don't use vckpg you may have to change more in `CMakeLists.txt` to build the program.

When you have the dependencies installed, you can run `gen_chap4.sh`.
It will build the approach,
train it from scratch, run all evaluations for the results chapter
and produce the corresponding figures.
```
$ bash gen_chap4.sh
```

## Formatting

Format with `ruff` and `clang-format`:
```
$ ruff format && clang-format -i -style=file src/*.cpp src/*.h
```