cmake_minimum_required(VERSION 3.20.0)
set(CMAKE_TOOLCHAIN_FILE "/home/jonas/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "-O0")

project("WarehouseSim")

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(simulation src/simulation_pybind11.cpp src/simulation.cpp src/agent.cpp)
install(TARGETS simulation DESTINATION .)