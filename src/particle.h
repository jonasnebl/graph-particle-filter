#ifndef WAREHOUSESIM_SRC_PARTICLE_H
#define WAREHOUSESIM_SRC_PARTICLE_H

#include <array>
#include <random>
#include <vector>

#include "warehouse_data.h"

using Point = std::pair<double, double>;

class Particle {
   public:
    Particle(graph_struct* graph_);                    // random particle
    Particle(const Particle& p);                       // copy constructor
    Particle(int edge_, double t, graph_struct* graph_);  // new custom particle
    Point get_position();
    double get_heading();
    double distance(Point robot_position, Point measured_position, double measured_heading);
    double likelihood_no_perception(std::vector<Point> robot_position);
    void predict(double T_step);
    int get_random_successor_edge(int edge);
    double get_random_time_of_edge_change(int edge);
    bool is_human_on_edge(int edge_input) const;
    double assignment_cost(Point p, double heading);

   private:
    // particle state
    int edge;
    double time_of_edge_change;
    double time_since_edge_change;

    // pointers to graph data
    graph_struct* graph;

    // random number generator
    std::mt19937 mt;
};

#endif