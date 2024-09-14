#ifndef WAREHOUSESIM_SRC_PARTICLE_H
#define WAREHOUSESIM_SRC_PARTICLE_H

#include <vector>
#include <array>
#include <random>

using Point = std::pair<double, double>;

struct Human {
    int edge;
    int next_edge;
    int previous_edge;
    double time_since_edge_change;
    double time_of_edge_change;
};

class Particle{
    public:
        // random initialization
        Particle(int N_humans_max, 
                 std::vector<Point>* nodes,
                 std::vector<std::pair<int,int>>* edges,
                 std::vector<std::vector<Point>>* racks,
                 std::vector<std::vector<int>>* successor_edges, 
                 std::vector<std::vector<std::array<double, 3>>>* pred_model_params);  
        Particle(const Particle &p); // copy initialization
        std::vector<std::tuple<Point, double, double>> simulate_measurement(Point robot_position);
        void predict(double T_step);
        int get_random_successor_edge(int edge);
        double get_random_time_of_edge_change(int current_edge, int next_edge);
        void update_params(double distance_metric);
        int find_edge_relative_index(int edge, int next_edge) const;
        std::vector<Human> humans;
    private:
        const std::vector<Point>* nodes;
        const std::vector<std::pair<int,int>>* edges;
        const std::vector<std::vector<Point>>* racks;
        const std::vector<std::vector<int>>* successor_edges;
        std::vector<std::vector<std::array<double, 3>>>* pred_model_params;        
        std::mt19937 mt;
};

#endif