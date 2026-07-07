#include "network.hpp"
#include <algorithm>
#include <utility>
#include <random>

Network::Network(std::vector<int> _sizes) : sizes(std::move(_sizes)) {
    num_layers = static_cast<int>(sizes.size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    biases.reserve(static_cast<size_t>(num_layers) - 1);
    weights.reserve(static_cast<size_t>(num_layers) - 1);

    for (size_t y = 1; y < static_cast<size_t>(num_layers); ++y) {
        int layer_sz = sizes[y];
        std::vector<double> layer_biases;
        layer_biases.reserve(static_cast<size_t>(layer_sz));
        for (int i = 0; i < layer_sz; ++i)
            layer_biases.emplace_back(dist(gen));
        biases.emplace_back(std::move(layer_biases));
    }
    
    for (size_t y = 1; y < static_cast<size_t>(num_layers); ++y) {
        int now_sz = sizes[y]; // W @ a => string in W = weights of the last layer for this neuron
        int last_layer = sizes[y - 1];

        std::vector<std::vector<double>> layer_weights;
        layer_weights.reserve(static_cast<size_t>(now_sz));
        
        for (int i = 0; i < now_sz; ++i) {
            std::vector<double> neuron_weights;
            neuron_weights.reserve(static_cast<size_t>(last_layer));
            for (int j = 0; j < last_layer; ++j)
                neuron_weights.emplace_back(dist(gen));
            layer_weights.emplace_back(std::move(neuron_weights));
        }
        weights.emplace_back(std::move(layer_weights));
    }
}
