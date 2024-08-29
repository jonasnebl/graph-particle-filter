#include "simulation.h"
#include "utils.h"

PYBIND11_MODULE(simulation, m) {
    m.doc() = "2D warehouse simulation";

    pybind11::class_<Simulation>(m, "Simulation")
        .def(pybind11::init<double, int, int>(), "Initialize the simulation",
             pybind11::arg("T_step"), pybind11::arg("N_humans"),
             pybind11::arg("N_robots"))
        .def("step", &Simulation::step, "Perform a simulation step",
             pybind11::arg("N_steps"))
        .def_readwrite("T_step", &Simulation::_T_step, "Time step of the simulation")
        .def_readwrite("N_humans", &Simulation::_N_humans, "Number of humans in the simulation")
        .def_readwrite("N_robots", &Simulation::_N_robots, "Number of robots in the simulation");
}