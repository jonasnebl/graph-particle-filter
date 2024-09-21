#include "particleTracker.h"
#include "simulation.h"

PYBIND11_MODULE(cpp_utils, m) {
    m.doc() = "cpp utils for the simulation and the tracker";

    pybind11::class_<Simulation>(m, "Simulation")
        .def(pybind11::init<double, int, int>(),
             "Initialize the "
             "simulation",
             pybind11::arg("T_step"), pybind11::arg("N_humans"), pybind11::arg("N_robots"))
        .def("step", &Simulation::step, "Perform a simulation step", pybind11::arg("N_steps"))
        .def_readwrite("T_step", &Simulation::_T_step, "Time step of the simulation")
        .def_readwrite("N_humans", &Simulation::_N_humans, "Number of humans in the simulation")
        .def_readwrite("N_robots", &Simulation::_N_robots, "Number of robots in the simulation");

    pybind11::class_<ParticleTracker>(m, "ParticleTracker")
        .def(pybind11::init<double, int, int>(), "Init the ParticleTracker",
             pybind11::arg("T_step"), pybind11::arg("N_humans_max"), pybind11::arg("N_particles"))
        .def("add_observation", &ParticleTracker::add_observation, "Add robot oservations",
             pybind11::arg("robot_perceptions"))
        .def("predict", &ParticleTracker::predict, "Predict by one T_step")
        .def("save_pred_model_params", &ParticleTracker::save_pred_model_params,
             "Save the prediction model parameters");
}
