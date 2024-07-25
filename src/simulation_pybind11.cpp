#include "simulation.h"

PYBIND11_MODULE(simulation, m) {
    m.doc() = "2D warehouse simulation";

    py::class_<Simulation>(m, "Simulation")
        .def(py::init<double, int, int>(), "Initialize the simulation",
             py::arg("T_step"), py::arg("N_humans"), py::arg("N_robots"))
        .def("step", &Simulation::step, "Perform a simulation step",
             py::arg("N_steps"));
}