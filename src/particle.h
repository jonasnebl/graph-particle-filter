#ifndef WAREHOUSESIM_SRC_PARTICLE_H
#define WAREHOUSESIM_SRC_PARTICLE_H

#include <vector>
#include <array>
#include <random>

using Point = std::pair<double, double>;

struct Human {
    int edge;
    int next_edge;
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