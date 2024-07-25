#ifndef AGENT_H
#define AGENT_H

#include "simulation.h"
#include <pybind11/pybind11.h>
#include <queue>

namespace py = pybind11;

class Simulation;

class Agent {
    public:
        Agent(double T_step, Simulation* simulation);
        void step();
        virtual py::dict log_state() = 0;
    protected:
        Simulation* _simulation;
        double _T_step;
        std::array<double, 3> pose;
        std::queue<std::size_t> path;
        void add_new_job_to_queue(std::size_t start_node);
};

#endif