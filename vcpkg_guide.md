# Vcpkg Guide

This guide provides the bash commands on how to download and install vcpkg, and use it to install `pybind11` and `libhungarian` for the WarehouseSim project.

## Disclaimer

This guide is meant as a help to speed up the installation process.

Please note that these instructions are not guaranteed to work on every machine due to unforseeable differences in system configurations, environments and versions, but rather provide help with the general process.

**Note:** 

I also had to use an older version of vcpkg to make it work during testing on my machine, 
that's why the checkout of a specific commit hash is added in the commands below.

Please ask me (`jonas.nebl@tum.de`) if you encounter any issues.


## Steps

1. **Download and install vcpkg**

   ```sh
   git clone https://github.com/microsoft/vcpkg.git
   cd vcpkg
   git checkout f56238700757aa05975e41fa835739c632810f3f
   ./bootstrap-vcpkg.sh
   ./vcpkg integrate install


2. **Install pybind11 and libhungarian**

    ```sh
   ./vcpkg install pybind11
   ./vcpkg install hungarian

2. **Adapt the `CMAKE_TOOLCHAIN_FILE` in `CMakeLists.txt`**

    ```
   set(CMAKE_TOOLCHAIN_FILE "/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake")
    ```
