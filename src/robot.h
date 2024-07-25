#ifndef ROBOT_H
#define ROBOT_H

#include "agent.h"
#include "simulation.h"

class Robot : public Agent {
    public:
        Robot(double T_step, Simulation* simulation) : Agent(T_step, simulation) {}
        virtual py::dict log_state() override;
};

#endif