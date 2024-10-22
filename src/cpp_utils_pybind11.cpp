#include "particleTracker.h"
#include "simulation.h"

PYBIND11_MODULE(cpp_utils, m) {
    m.doc() = "cpp utils for the simulation and the tracker";

    pybind11::class_<Simulation>(m, "Simulation")
        .def(pybind11::init<double, int, int>(),
             "Initialize the "
             "simulation",
             pybind11::arg("T_step"), pybind11::arg("N_humans"), pybind11::arg("N_robots"))
        .def("step", &Simulation::step, "Perform a simulation step", pybind11::arg("N_steps"));

    pybind11::class_<ParticleTracker>(m, "ParticleTracker")
        .def(pybind11::init<double, int, int>(), "Init the ParticleTracker",
             pybind11::arg("T_step"), pybind11::arg("N_humans_max"), pybind11::arg("N_particles"))
        .def("merge_perceptions", &ParticleTracker::merge_perceptions,
             pybind11::arg("robot_perceptions"), "Merge perceptions from different robots")
        .def("calc_assignment_cost_matrix", &ParticleTracker::calc_assignment_cost_matrix,
             "Calculate the assignment cost matrix", pybind11::arg("perceived_humans"),
             pybind11::arg("particle_index"))
        .def("add_observation", &ParticleTracker::add_observation, "Add robot oservations",
             pybind11::arg("robot_perceptions"))
        .def("predict", &ParticleTracker::predict, "Predict by one T_step");
}
