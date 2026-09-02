#include <vector>
#include <stdexcept>

#include "network.hpp"
#include "test_utlis.hpp"
#include "matrix.hpp"

int main() {
    // init network
    std::vector<int> sizes0 = {1, 2, 3, 4, 5};
    Network network1(sizes0);

    std::vector<Matrix>& biases1 = network1.getBiasesRef();
    std::vector<Matrix>& weights1 = network1.getWeightsRef();
    std::vector<int>& sizes1 = network1.getSizesRef();
    Matrix input1(1, 1, std::vector<Scalar>{1});

    // check std init errors
    CHECK_THROWS_AS(Network({}), std::invalid_argument);
    CHECK_THROWS_AS(Network({1}), std::invalid_argument);
    CHECK_THROWS_AS(Network({-1, -2}), std::invalid_argument);

    // check if vector of the sizes is correct
    CHECK(sizes1[0] == sizes0[0]);
    CHECK(sizes1[1] == sizes0[1]);
    CHECK(sizes1[2] == sizes0[2]);
    CHECK(sizes1[3] == sizes0[3]);
    CHECK(sizes1[4] == sizes0[4]);

    // check if matrixes of the weight is correct
    CHECK(weights1[0].getRows() == sizes0[1]);
    CHECK(weights1[0].getColumns() == sizes0[0]);
    CHECK(weights1[1].getRows() == sizes0[2]);
    CHECK(weights1[1].getColumns() == sizes0[1]);
    CHECK(weights1[2].getRows() == sizes0[3]);
    CHECK(weights1[2].getColumns() == sizes0[2]);
    CHECK(weights1[3].getRows() == sizes0[4]);
    CHECK(weights1[3].getColumns() == sizes0[3]);

    // check if matrixes of the biases is correct
    CHECK(biases1[0].getRows() == sizes0[1]);
    CHECK(biases1[0].getColumns() == 1);
    CHECK(biases1[1].getRows() == sizes0[2]);
    CHECK(biases1[1].getColumns() == 1);
    CHECK(biases1[2].getRows() == sizes0[3]);
    CHECK(biases1[2].getColumns() == 1);
    CHECK(biases1[3].getRows() == sizes0[4]);
    CHECK(biases1[3].getColumns() == 1);

    // just check if does not have error with activation function
    network1.feedforward(input1);

    // check feedforward without activation function
    std::vector<int> sizes2 = {3, 2, 4, 3};
    Matrix input2(3, 1, {1, 2, 3});
    Matrix matrix2_1(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix matrix2_2(4, 2, {1, 2, 3, 4, 5, 6, 7, 8});
    Matrix matrix2_3(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Matrix bias2_1(2, 1, {1, 2});
    Matrix bias2_2(4, 1, {1, 2, 3, 4});
    Matrix bias2_3(3, 1, {1, 2, 3});

    std::vector<Matrix> weights2 = {matrix2_1, matrix2_2, matrix2_3};
    std::vector<Matrix> biases2 = {bias2_1, bias2_2, bias2_3};

    Network network2(sizes2, biases2, weights2);
    Matrix matrix4(3, 1, {2821, 6542, 10263});
    CHECK(network2.feedforward(input2, "", false) == matrix4);

    // check random backprop (just if works, no deterministic result check, which i have to add) 
    Network network3({784, 30, 10});
    Matrix x3(784, 1), y3(10, 1);
    auto tmp3 = network3.backprop(x3, y3);

    // need to determenistic check if backprop really works


    // test random update_mini_batch
    Network network4({784, 30, 10});
    Matrix x4(784, 1), y4(10, 1);
    //network3.update_mini_batch(std::vector<std::pair<Matrix, Matrix>>(std::pair<Matrix, Matrix>(x4, y4)), Scalar(3.0));
    network3.update_mini_batch({{x4, y4}}, Scalar(3.0));


    // need to determenistic check if update_mini_batch really works

    
    printResults();
    return failed;
}
