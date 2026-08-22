#include <stdexcept>

#include "test_utlis.hpp"
#include "activation_functions.hpp"
//#include "matrix.hpp"


int main() {
    CHECK(sigmoid(0) == 0.5);
    CHECK(sigmoid_prime(0) == 0.25);

    printResults();
    return failed;
}