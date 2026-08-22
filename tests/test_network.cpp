#include <vector>
#include <stdexcept>

#include "network.hpp"
#include "test_utlis.hpp"
#include "matrix.hpp"

int main() {
    // init network
    std::vector<int> sizes = {1, 2, 3, 4, 5};
    Network network1(sizes);

    std::vector<Matrix>& biases1 = network1.getBiasesRef();
    std::vector<Matrix>& weights1 = network1.getWeightsRef();
    std::vector<int>& sizes1 = network1.getSizesRef();

    // check if vector of the sizes is correct
    CHECK(sizes1[0] == sizes[0]);
    CHECK(sizes1[1] == sizes[1]);
    CHECK(sizes1[2] == sizes[2]);
    CHECK(sizes1[3] == sizes[3]);
    CHECK(sizes1[4] == sizes[4]);

    // check if matrixes of the weight is correct
    CHECK(weights1[0].getRows() == sizes[1]);
    CHECK(weights1[0].getColumns() == sizes[0]);
    CHECK(weights1[1].getRows() == sizes[2]);
    CHECK(weights1[1].getColumns() == sizes[1]);
    CHECK(weights1[2].getRows() == sizes[3]);
    CHECK(weights1[2].getColumns() == sizes[2]);
    CHECK(weights1[3].getRows() == sizes[4]);
    CHECK(weights1[3].getColumns() == sizes[3]);

    //check if matrixes of the biases is correct
    CHECK(biases1[0].getRows() == sizes[1]);
    CHECK(biases1[0].getColumns() == 1);
    CHECK(biases1[1].getRows() == sizes[2]);
    CHECK(biases1[1].getColumns() == 1);
    CHECK(biases1[2].getRows() == sizes[3]);
    CHECK(biases1[2].getColumns() == 1);
    CHECK(biases1[3].getRows() == sizes[4]);
    CHECK(biases1[3].getColumns() == 1);

    printResults();
    return failed;
}
