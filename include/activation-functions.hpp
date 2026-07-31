#pragma once

#include "types.hpp"
#include <cmath>

inline Scalar sigmoid(Scalar z) {
    static_cast<Scalar>(1) / (static_cast<Scalar>(1) + std::exp(-z));
}
inline Scalar sigmoid_prime(Scalar z) {
    Scalar sigmoid_value = sigmoid(z);
    return sigmoid_value * (static_cast<Scalar>(1) - sigmoid_value);
}
