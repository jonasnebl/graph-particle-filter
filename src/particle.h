#ifndef WAREHOUSESIM_SRC_PARTICLE_H
#define WAREHOUSESIM_SRC_PARTICLE_H

#include <array>
#include <random>
#include <vector>

#include "warehouse_data.h"

using Point = std::pair<double, double>;

class Particle {
   public:
    Particle(graph_struct* graph_);
    Point get_position();
    double get_heading();
    double distance(Point robot_position, Point measured_position, double measured_heading);
    double likelihood_no_perception(std::vector<Point> robot_position);
    void rewrite_from_perception(Point perceived_pos, double position_stddev,
                                 double perceived_heading, double heading_stddev);
    void rewrite_from_other_particle(const Particle& p);
    static std::tuple<int, double> get_belonging_edge(Point position, double heading,
                                                      const graph_struct& graph);
    static std::tuple<double, double> edge_to_pose_distance_and_t(int edge, Point position,
                                                                  double heading,
                                                                  const graph_struct& graph);
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