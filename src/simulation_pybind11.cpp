#include "simulation.h"

PYBIND11_MODULE(simulation, m) {
    m.doc() = "2D warehouse simulation";

    pybind11::class_<Simulation>(m, "Simulation")
        .def(pybind11::init<double, int, int>(), "Initialize the simulation",
             pybind11::arg("T_step"), pybind11::arg("N_humans"),
             pybind11::arg("N_robots"))
        .def("step", &Simulation::step, "Perform a simulation step",
             pybind11::arg("N_steps"));
}