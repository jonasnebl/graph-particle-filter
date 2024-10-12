#ifndef WAREHOUSESIM_SRC_PARTICLE_H
#define WAREHOUSESIM_SRC_PARTICLE_H

#include <vector>
#include <array>
#include <random>

using Point = std::pair<double, double>;

class Particle{
    public:
        // random initialization
        Particle(std::vector<Point>* nodes,
                 std::vector<std::pair<int,int>>* edges,
                 std::vector<std::vector<Point>>* racks,
                 std::vector<std::vector<int>>* successor_edges, 
                 std::vector<std::vector<std::array<double, 3>>>* pred_model_params);  
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
        int find_edge_relative_index(int edge, int next_edge) const;
        bool is_human_on_edge(int edge_input) const;
    private:
        int edge;
        int next_edge;
        double time_of_edge_change;
        double time_since_edge_change;
        double previous_distance = 10000;

        const std::vector<Point>* nodes;
        const std::vector<std::pair<int,int>>* edges;
        const std::vector<std::vector<Point>>* racks;
        const std::vector<std::vector<int>>* successor_edges;
        std::vector<std::vector<std::array<double, 3>>>* pred_model_params;        
        
        std::mt19937 mt;
};

#endif