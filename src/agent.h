#ifndef AGENT_H
#define AGENT_H

#include "simulation.h"
#include <pybind11/pybind11.h>
#include <deque>

namespace py = pybind11;

class Simulation;

class Agent {
    public:
        Agent(double T_step, bool is_human, Simulation* simulation);
        bool _is_human;
        void step();
        py::dict log_state();
    protected:
        Simulation* _simulation;
        double _T_step;
        std::array<double, 3> pose;
        double speed = 0.5;
        std::deque<std::pair<double, double>> path;
        void add_new_job_to_deque();
        void add_node_to_deque(std::size_t node_index);
        std::size_t DROPOFF_HUMANS = 1;
        std::size_t DROPOFF_ROBOTS = 2;
        std::vector<std::pair<double, double>> perceive_humans();
        std::vector<std::pair<double, double>> extend_perception(std::vector<std::pair<double, double>> perceived_humans);
        bool do_intersect(std::pair<double, double> p1, 
                          std::pair<double, double> q1, 
                          std::pair<double, double> p2, 
                          std::pair<double, double> q2);
};

#endif