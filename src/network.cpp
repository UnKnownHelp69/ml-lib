#include "network.hpp"
#include "types.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

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
        throw std::invalid_argument("Size of network input does not match with right.");
    if (a.getColumns() != 1)
        throw std::invalid_argument("Matrix of network input have to be an vector, not matrix.");

    for (size_t i = 0; i < num_layers - 1; ++i) {
        a = weights[i] * a + biases[i];
        if (ifApplyActivation)
            a.applyActivation(nameActivationFunction);
    }

    return a;
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> Network::backprop(Matrix& x, Matrix& y) {
    /*
    Get x (784, 1) - input data, y (10, 1) - real answer
    
    Return pair of nablaB, nablaW, representing the gradient 
    for the cost function C_x. nablaB and nablaW are vectors 
    of Matrix, similar to biases and weights
    */

    // NB I am USING METHODS of MATRIX, SO IT CAN CAUSE UB IF THERE IS SMTH NEED TO BE AS WAS BEFORE METHOD 
    // (LIKE ACTIVATION FUNCTION AND TRANSPOSE)
    size_t numWeights = num_layers - 1;
    std::vector<Matrix> nablaB, nablaW;
    nablaB.reserve(numWeights);
    nablaW.reserve(numWeights);
    for (size_t i = 0; i < numWeights;  ++i) {
        nablaB.emplace_back(biases[i].getRows(), biases[i].getColumns()).zeros();
        nablaW.emplace_back(weights[i].getRows(), weights[i].getColumns()).zeros();
    }

    // feedforward
    Matrix activation = x;
    std::vector<Matrix> activations = {activation};
    std::vector<Matrix> zs = {};
    for (size_t i = 0; i < numWeights; ++i) {
        Matrix b = biases[i], w = weights[i];
        auto z = (w * activation) + b;
        zs.push_back(z);
        z.applyActivation(); // it is ok right there, z cause'll change in the future iteration
        activation = z;
        activations.push_back(activation);
    }
    // backward pass
    auto tmp1 = zs[zs.size() - 1]; // for now it is tmp args
    tmp1.applyActivation("sigmoid_prime");
    auto delta = cost_derivative(activations[activations.size() - 1], y) % tmp1;
    nablaB[nablaB.size() - 1] = delta;

    auto tmp2 = activations[activations.size() - 2];
    tmp2.transpose();
    nablaW[nablaW.size() - 1] = delta * tmp2;

    for (size_t l = 2; l < num_layers; ++l) {
        auto sp = zs[zs.size() - l]; 
        sp.applyActivation("sigmoid_prime"); // it is ok right there
        auto tmp3 = weights[weights.size() - l + 1];
        tmp3.transpose();
        delta = (tmp3 * delta) % sp;
        nablaB[nablaB.size() - l] = delta;
        auto tmp4 = activations[activations.size() - l - 1];
        tmp4.transpose();
        nablaW[nablaW.size() - l] = delta * tmp4;
    }

    return {nablaB, nablaW};
}

void Network::update_mini_batch(std::vector<std::pair<Matrix, Matrix>>& mini_batch, 
        Scalar eta) {
    /*
    Update the network weights and biases, using gradient descent by 
    backpropagation and apply it to mini_batch. 
    
    mini_batch is a pair of Matrixes (x, y)
    eta is a learning rate
    */
    size_t numWeights = num_layers - 1;
    std::vector<Matrix> nabla_b, nabla_W;
    nabla_b.reserve(numWeights);
    nabla_W.reserve(numWeights);
    for (size_t i = 0; i < numWeights; ++i) {
        nabla_b.emplace_back(biases[i].getRows(), biases[i].getColumns()).zeros();
        nabla_W.emplace_back(weights[i].getRows(), weights[i].getColumns()).zeros();
    }
    size_t miniBatchSize = mini_batch.size(); 

    for (size_t i = 0; i < miniBatchSize; ++i) {
        Matrix& x = mini_batch[i].first;
        Matrix& y = mini_batch[i].second;
        auto tmp = backprop(x, y);
        std::vector<Matrix>& delta_nabla_b = tmp.first;
        std::vector<Matrix>& delta_nabla_W = tmp.second;
        for (size_t j = 0; j < numWeights; ++j) {
            nabla_b[j] = nabla_b[j] + delta_nabla_b[j]; // need to change on += (now matrix has not this operator)
            nabla_W[j] = nabla_W[j] + delta_nabla_W[j];
        }
    }
    for (size_t i = 0; i < numWeights; ++i) {
        weights[i] = weights[i] - nabla_W[i] * (eta / static_cast<Scalar>(miniBatchSize));
        biases[i] = biases[i] - nabla_b[i] * (eta / static_cast<Scalar>(miniBatchSize));
    }
}

void Network::SGD(std::vector<std::pair<Matrix, Matrix>>& trainingData, size_t epochs, 
        size_t mini_batch_size, Scalar eta, std::vector<std::pair<Matrix, Matrix>>& testData) {
    /*
    place for the comments, BUT I DONT HAVE THE MAIN PART OF RANDOM SHUFFLING - ADDD IT !!!!
    */

    size_t n = trainingData.size();
    for (size_t j = 0; j < epochs; ++j) {
        //random.shuffle(training_data)
        for (size_t k = 0; k < n; k += mini_batch_size) {
            std::vector<std::pair<Matrix, Matrix>> miniBatch;
            miniBatch.reserve(mini_batch_size);
            for (size_t i = k; i < k + mini_batch_size; ++i) {
                miniBatch.emplace_back(trainingData[i].first, trainingData[i].second);
            }
            update_mini_batch(miniBatch, eta);
        }
        std::cout << "Epoch {" << j << "} is completed.\n";
    }

    if (!testData.size()) {
        std::cout << "Nothing to test\n";
    } else {
        // i need to add there evaluate implementation
        std::cout << "Not implemented, but will be soon\n";
    }
}
