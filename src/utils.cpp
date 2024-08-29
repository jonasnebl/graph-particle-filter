#include "utils.h"
#include "agent.h"
#include "warehouse_data.h"

Utils::Utils(double T_step, int N_humans, int N_robots) {
    // load graph
    nodes = warehouse_data::nodes;
    edges = warehouse_data::edges;
    edge_weights = warehouse_data::edge_weights;
    racks = warehouse_data::racks;
    node_polygons = warehouse_data::node_polygons;
}

std::vector<std::pair<double, double>> Utils::get_belonging_node(Point robot_position, 
                                                                     std::vector<Point> perceived_human_positions) {
    std::vector<std::pair<double, double>> result(nodes.size());
    std::fill(result.begin(), result.end(),
              std::make_pair<double, double>(0.0, 0.0));

    for (int i = 0; i < nodes.size(); i++) {
        // 1. Calculate confidence for each node by evaluating visibility
        result[i].second = static_cast<double>(
            Agent::check_viewline(robot_position, nodes[i], racks));

        // 2. Calculate probability of human presence at each node for each
        // human
        for (const auto &mean_pos_human : perceived_human_positions) {
            if (Agent::is_point_in_polygon(mean_pos_human, node_polygons[i])) {
                result[i].first = 1.0;
            }
        }
    }
    return result;
}
