cmake_minimum_required(VERSION 3.20.0)
set(CMAKE_TOOLCHAIN_FILE "/home/jonas/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_STANDARD 20)

project("WarehouseSim")

find_package(pybind11 CONFIG REQUIRED)
find_package(hungarian CONFIG REQUIRED)

pybind11_add_module(cpp_utils 
                    src/cpp_utils_pybind11.cpp 
                    src/simulation.cpp 
                    src/agent.cpp
                    src/particleTracker.cpp
                    src/particle.cpp)

install(TARGETS cpp_utils DESTINATION .)
target_link_libraries(cpp_utils PRIVATE hungarian::hungarian)