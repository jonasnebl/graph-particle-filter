#include "human.h"

py::dict Human::log_state() {
    py::dict state;
    state["x"] = pose[0];
    state["y"] = pose[1];
    state["theta"] = pose[2];
    return state;
}