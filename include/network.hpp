#pragma once

#include <vector>

class Network {
private:
    int num_layers;
    std::vector<int> sizes;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<std::vector<double>>> weights;
public:
    explicit Network(std::vector<int> _sizes);
};
