#ifndef WAREHOUSESIM_SRC_PARTICLE_H
#define WAREHOUSESIM_SRC_PARTICLE_H

#include <vector>
#include <array>
#include <random>

#include "warehouse_data.h"

using Point = std::pair<double, double>;

class Particle{
    public:
        // random initialization
        Particle(graph_struct* graph_,
                 std::vector<std::vector<std::array<double, 3>>>* pred_model_params_);  
        Particle(const Particle &p); // copy constructor
        Particle(int edge_, double t, const Particle& p); // copy constructor with custom edge
        Point get_position();
        double get_heading();
        double distance(Point robot_position, Point measured_position, double measured_heading);
        double likelihood_no_perception(std::vector<Point> robot_position);
        double measurement_noise_pdf(Point particle_position, Point measured_position);
        void predict(double T_step);
        int get_random_successor_edge(int edge);
        double get_random_time_of_edge_change(int current_edge, int next_edge);
        bool is_human_on_edge(int edge_input) const;
        double distance(Point p, double heading);
    private:
        // particle state
        int edge;
        int next_edge;
        double time_of_edge_change;
        double time_since_edge_change;

        // pointers to graph data
        graph_struct* graph;
        std::vector<std::vector<std::array<double, 3>>>* pred_model_params;        
        
        std::mt19937 mt;
};

#endif