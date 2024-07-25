#ifndef GRAPH_DATA_H
#define GRAPH_DATA_H

#include <vector>
namespace graph_data {
    std::vector<std::pair<double, double>> nodes = {
        {0, 0}, 
        {0, 1}, 
        {1, 1}, 
        {0.5, 1.7}, 
        {2, 1}, 
        {2, 2}, 
        {3, 0.3}
    };
    std::vector<std::pair<std::size_t, std::size_t>> edges = {
        {0, 1}, 
        {0, 2}, 
        {1, 2}, 
        {1, 3}, 
        {2, 3}, 
        {3, 2}, 
        {2, 4}, 
        {3, 5}, 
        {5, 4}, 
        {4, 6}, 
        {5, 6}
    };
    std::vector<double>edge_weights = {3, 4, 3, 3, 2, 2, 3, 4, 3, 4, 6};
}

#endif