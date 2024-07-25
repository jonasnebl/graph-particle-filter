#ifndef HUMAN_H
#define HUMAN_H

#include "agent.h"

class Human : public Agent {
    public:
        Human(double T_step, Simulation* simulation) : Agent(T_step, simulation) {}
        virtual py::dict log_state() override;
};

#endif