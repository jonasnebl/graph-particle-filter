#ifndef WAREHOUSESIM_SRC_AGENT_H
#define WAREHOUSESIM_SRC_AGENT_H

#include <random> 
#include <deque>

#include <pybind11/pybind11.h>

using Point = std::pair<double, double>;
class Simulation;

class Agent {
  public:
    Agent(double T_step, bool is_human, Simulation *simulation);
    bool _is_human;
    void step();
    pybind11::dict log_state();
    int get_belonging_edge();
  
    static double distance(Point p1, Point p2);
    static bool do_intersect(Point p1, Point q1, Point p2, Point q2);
    static bool check_viewline(Point pos1, Point pos2, 
                               std::vector<std::vector<Point>> racks);
    static bool is_point_in_polygon(Point point, std::vector<Point> polygon);

  protected:
    Simulation *_simulation;
    double _T_step;

    Point position;
    double heading = 0;
    double velocity = 2;
    std::deque<Point> path;

    std::vector<pybind11::dict> perceive_humans();

    void add_new_job_to_deque();
    void add_node_to_deque(int node_index);
    int DROPOFF_HUMANS = 1;
    int DROPOFF_ROBOTS = 2;

    std::mt19937 mt;
    std::normal_distribution<double> position_noise;
    std::normal_distribution<double> heading_noise;
    std::normal_distribution<double> velocity_noise;
};

#endif