#ifndef WAREHOUSESIM_SRC_AGENT_H
#define WAREHOUSESIM_SRC_AGENT_H

#include <pybind11/pybind11.h>

#include <deque>
#include <random>

#include "warehouse_data.h"

class Simulation;

using Point = std::pair<double, double>;

const double D_MIN = 7.5;
const double D_MAX = 15.0;
const double DETECTION_PROBABILITY_IN_RANGE = 0.95;
const double XY_STDDEV = 0.02;
const double HEADING_STDDEV = 15.0 * M_PI / 180.0;
const double ROBOT_VELOCITY = 2.2;
const double HUMAN_VELOCITY_MEAN = 1.3;
const double HUMAN_VELOCITY_STDDEV = 0.1;

const double PAUSE_VELOCITY = 0.1;
const double OUT_OF_WAREHOUSE_VELOCITY =
    0.0006;  // the smaller, the longer the agent stays out of the warehouse
const double LEAVE_WAREHOUSE_PROBABILITY = 0.02;

const double SMOOTHING_STRENGTH = 0.035;
const double SMOOTHING_ITERATIONS = 5;

enum class AgentType { HUMAN, ROBOT };

class Agent {
   public:
    Agent(double T_step, AgentType type_, Simulation* simulation_);
    AgentType type;
    void step();
    pybind11::dict log_state();
    static bool check_viewline(Point pos1, Point pos2, std::vector<std::vector<Point>> racks);
    static double manhattan_distance(Point p1, Point p2);
    static double euclidean_distance(Point p1, Point p2);
    static double probability_in_viewrange(double dist);

   private:
    Simulation* simulation;
    double _T_step;
    std::mt19937 mt;

    Point position;
    double heading = 0.0;
    double velocity = 0.0;
    std::normal_distribution<double> velocity_distribution;
    double get_random_velocity();

    std::deque<std::pair<Point, double>> path;  // (next position, velocity to this position)
    int current_final_path_node;                // holds node index of last node in path

    std::vector<pybind11::dict> perceive_humans();
    std::normal_distribution<double> position_noise;
    std::normal_distribution<double> heading_noise;

    void add_new_double_cycle_to_deque();
    void add_path_to_deque(std::vector<int> path_to_target, double path_velocity);
    void add_node_to_deque(int node_index, double path_velocity);
    void smooth_path(int start, int end, double strength);
    int random_staging_node();
    int random_storage_node();
    bool random_leave_warehouse();
    static bool do_intersect(Point p1, Point q1, Point p2, Point q2);
    static bool is_point_in_polygon(Point point, std::vector<Point> polygon);
    bool random_check_viewrange(Point pos1, Point pos2);
};

#endif  // WAREHOUSESIM_SRC_AGENT_H