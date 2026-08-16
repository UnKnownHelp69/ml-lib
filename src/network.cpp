#include "network.hpp"
#include "types.hpp"

#include <algorithm>
#include <utility>
#include <random>
#include <stdexcept>

Network::Network(std::vector<int> _sizes) : num_layers(_sizes.size()), sizes(std::move(_sizes)) {
    if (sizes.empty()) 
        throw std::invalid_argument("Network must has at least one laywer.");

    for (int& i : sizes) {
        if (i <= 0) 
            throw std::invalid_argument("Laywer size must be positive.");
    }

    size_t num_weights = num_layers - 1;

    biases.reserve(num_weights);
    weights.reserve(num_weights);

    for (size_t i = 1; i < num_layers; ++i) { // first is always input
        // matrix of weights is must be in the left of the matrix results when multiply
        biases.emplace_back(static_cast<size_t>(sizes[i]), 1);
        weights.emplace_back(static_cast<size_t>(sizes[i]), static_cast<size_t>(sizes[i - 1]));
    }

}
