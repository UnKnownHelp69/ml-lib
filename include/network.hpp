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
    explicit Network(std::vector<int> _sizes, std::vector<Matrix> biases, std::vector<Matrix> weights);

    std::vector<Matrix>& getBiasesRef() { return biases; }
    std::vector<Matrix>& getWeightsRef() { return weights; }
    std::vector<int>& getSizesRef() { return sizes; }

    Matrix feedforward(Matrix a, std::string_view nameActivationFunction = "sigmoid", 
            bool ifApplyActivation = true);
};
