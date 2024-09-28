#ifndef WAREHOUSESIM_SRC_AGENT_H
#define WAREHOUSESIM_SRC_AGENT_H

#include <random> 
#include <deque>

#include <pybind11/pybind11.h>

class Simulation;

using Point = std::pair<double, double>;

const double D_MIN = 10.0;
const double D_MAX = 20.0;
const double DETECTION_PROBABILITY_IN_RANGE = 0.99;
const double XY_STDDEV = 0.02;
const double HEADING_STDDEV = 15 * 3.14159 / 180;
const double ROBOT_VELOCITY = 2.2;
const double HUMAN_VELOCITY_MEAN = 1.4;
const double HUMAN_VELOCITY_STDDEV = 0.2;

class Agent {
  public:
    Agent(double T_step, bool is_human, Simulation *simulation);
    bool _is_human;
    void step();
    pybind11::dict log_state();
    int get_belonging_edge();
    static bool check_viewline(Point pos1, Point pos2, 
                               std::vector<std::vector<Point>> racks); 
    static double manhattan_distance(Point p1, Point p2);

  private:
    Simulation *_simulation;
    double _T_step;
    std::mt19937 mt;

    Point position;
    double heading = 0;
    double velocity = 2;
    std::normal_distribution<double> velocity_distribution;
    double get_random_velocity();
   
    std::deque<std::pair<Point, double>> path; // (next position, velocity to this position)
    int current_final_path_node; // holds node index of last node in path

    std::vector<pybind11::dict> perceive_humans();
        std::normal_distribution<double> position_noise;
    std::normal_distribution<double> heading_noise;

    void add_new_double_cycle_to_deque();
    void add_node_to_deque(int node_index, double path_velocity);
    void smooth_path(int start, int end, double strength);
    int random_staging_node();
    int random_storage_node();
    std::discrete_distribution<> staging_node_distribution;
    std::discrete_distribution<> storage_node_distribution;

    static double euclidean_distance(Point p1, Point p2);
    static bool do_intersect(Point p1, Point q1, Point p2, Point q2);
    static bool is_point_in_polygon(Point point, std::vector<Point> polygon);
    bool random_check_viewrange(Point pos1, Point pos2);   
};

#endif