#include "particleTracker.h"
#include "simulation.h"

PYBIND11_MODULE(cpp_utils, m) {
    m.doc() = "cpp utils for the simulation and the tracker";

    pybind11::class_<Simulation>(m, "Simulation")
        .def(pybind11::init<double, int, int, bool>(), "Initialize the simulation",
             pybind11::arg("T_step"), pybind11::arg("N_humans"), pybind11::arg("N_robots"),
             pybind11::arg("allow_warehouse_leaving"))
        .def("step", &Simulation::step, "Perform simulation steps", pybind11::arg("N_steps"));

    pybind11::class_<ParticleTracker>(m, "ParticleTracker")
        .def(pybind11::init<double, int, int>(), "Init the ParticleTracker",
             pybind11::arg("T_step"), pybind11::arg("N_tracks"), pybind11::arg("N_particles"))
        .def("merge_perceptions", &ParticleTracker::merge_perceptions,
             pybind11::arg("robot_perceptions"), "Merge perceptions from different robots")
        .def("add_merged_perceptions", &ParticleTracker::add_merged_perceptions,
             "Add robot oservations", pybind11::arg("perceived_humans"),
             pybind11::arg("robot_positions"))
        .def("estimate_N_humans", &ParticleTracker::estimate_N_humans,
             "Estimate the number of humans", pybind11::arg("N_perceived"))
        .def("add_one_track", &ParticleTracker::add_one_track)
        .def("remove_one_track", &ParticleTracker::remove_one_track)
        .def("calc_training_data", &ParticleTracker::calc_training_data,
             "Calculate the training data for the edge changes and duration")
        .def("predict", &ParticleTracker::predict, "Predict by one T_step");

    pybind11::class_<Agent>(m, "Agent")
        .def_static("probability_in_viewrange", &Agent::probability_in_viewrange,
                    "Calculate the euclidean distance between two points", pybind11::arg("dist"));
}
