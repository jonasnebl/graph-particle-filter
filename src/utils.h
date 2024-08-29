#ifndef WAREHOUSESIM_SRC_UTILS_H
#define WAREHOUSESIM_SRC_UTILS_H

class Utils {
   public:
    Utils();

    // warehouse structure
    std::vector<Point> nodes;
    std::vector<std::pair<int, int>> edges;
    std::vector<double> edge_weights;
    std::vector<std::vector<Point>> racks;
    std::vector<std::vector<Point>> node_polygons;
    std::vector<Point> get_belonging_node(Point robot_position, 
                                          std::vector<Point> perceived_human_positions);
};

#endif