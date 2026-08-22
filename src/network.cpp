#include "network.hpp"
#include "types.hpp"

#include <algorithm>
#include <utility>
#include <random>
#include <stdexcept>

Network::Network(std::vector<int> _sizes) : num_layers(_sizes.size()), sizes(std::move(_sizes)) {
    if (sizes.empty() || sizes.size() == 1) 
        throw std::invalid_argument("Network must has at least one laywer.");

    for (int& i : sizes) {
        if (i <= 0) 
            throw std::invalid_argument("Laywer size must be positive.");
    }

    size_t num_weights = num_layers - 1;

    biases.reserve(num_weights);
    weights.reserve(num_weights);

    for (size_t i = 0; i < num_weights; ++i) { // first is always input
        // matrix of weights is must be in the left of the matrix results when multiply
        biases.emplace_back(static_cast<size_t>(sizes[i + 1]), 1);
        weights.emplace_back(static_cast<size_t>(sizes[i + 1]), static_cast<size_t>(sizes[i]));
    }
}


Network::Network(std::vector<int> _sizes, std::vector<Matrix> _biases, std::vector<Matrix> _weights) : 
    num_layers(_sizes.size()), sizes(std::move(_sizes)), biases(std::move(_biases)), weights(std::move(_weights)) {
        // this constructor for now is only to test with known network, so do no have any tests
        if (sizes.empty()) 
            throw std::invalid_argument("Network must has at least one laywer.");

        for (int& i : sizes) {
            if (i <= 0) 
                throw std::invalid_argument("Laywer size must be positive.");
        }

        size_t num_weights = num_layers - 1;

        if (biases.size() != num_weights)
            throw std::invalid_argument("Biases amount is not correct.");
        if (weights.size() != num_weights)
            throw std::invalid_argument("Weights amount is not correct.");

        for (size_t i = 0; i < num_weights; ++i) { // first is always input
            // matrix of weights is must be in the left of the matrix results when multiply
            if (biases[i].getRows() != static_cast<size_t>(sizes[i + 1]) 
                    || biases[i].getColumns() != 1)
                throw std::invalid_argument("Biases size is not correct.");
            if (weights[i].getRows() != static_cast<size_t>(sizes[i + 1]) 
                    || weights[i].getColumns() != static_cast<size_t>(sizes[i]))
                throw std::invalid_argument("Weights size is not correct.");           
        }
    }


Matrix Network::feedforward(Matrix a, std::string_view nameActivationFunction, 
            bool ifApplyActivation) {
    /* 
        Return the output of the network if a is input of the network (input is vector)
        nameActivationFunction and ifApplyActivation for now is only to test functionality
    */
    if (a.getRows() != weights[0].getColumns()) 
        throw std::invalid_argument("Size of network input does not match with right");
    if (a.getColumns() != 1)
        throw std::invalid_argument("Matrix of network input have to be an vector, not matrix");

    for (size_t i = 0; i < num_layers - 1; ++i) {
        a = weights[i] * a + biases[i];
        if (ifApplyActivation)
            a.applyActivation(nameActivationFunction);
    }

    return a;
}
