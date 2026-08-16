#pragma once

#include "matrix.hpp"

#include <vector>

class Network {
private:
    size_t num_layers;
    std::vector<int> sizes;
    std::vector<Matrix> biases;
    std::vector<Matrix> weights;
public:
    explicit Network(std::vector<int> _sizes); // forbid non implicit changes

    std::vector<Matrix>& getBiasesRef() { return biases; }
    std::vector<Matrix>& getWeightsRef() { return weights; }
    std::vector<int>& getSizesRef() { return sizes; }
};
