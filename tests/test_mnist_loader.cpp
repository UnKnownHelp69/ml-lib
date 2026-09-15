#include <string>
#include <stdexcept>
#include <utility>

#include "test_utlis.hpp"
#include "mnist_loader.hpp"

int main() {
    // check MNIST input correctness
    std::string trainPath = "../../MNIST/mnist_train.csv";
    std::string testPath = "../../MNIST/mnist_test.csv";
    MnistLoader mnistLoader0(trainPath, testPath);
    auto [training_data, validation_data, test_data] = mnistLoader0.load_data_wrapper();

    CHECK(training_data.size() == 48000);
    CHECK(validation_data.size() == 12000);
    CHECK(test_data.size() == 10000);

    // check errors throwing
    std::string failPath1 = "../../tests/MNIST/mnist_fail1.csv";
    std::string failPath2 = "../../tests/MNIST/mnist_fail2.csv";
    std::string failPath3 = "../../tests/MNIST/mnist_fail3.csv";

    MnistLoader mnistLoader1(failPath1, testPath);
    MnistLoader mnistLoader2(failPath2, testPath);
    MnistLoader mnistLoader3(failPath3, testPath);

    CHECK_THROWS_AS(mnistLoader1.load_data_wrapper(), std::runtime_error);
    CHECK_THROWS_AS(mnistLoader2.load_data_wrapper(), std::runtime_error);
    CHECK_THROWS_AS(mnistLoader3.load_data_wrapper(), std::runtime_error);

    // chech the invalid paths
    std::string failPath4 = "";
    std::string failPath5 = "../../tests/bam.csv";

    CHECK_THROWS_AS(MnistLoader mnistLoader4(failPath4, testPath), std::invalid_argument);
    CHECK_THROWS_AS(MnistLoader mnistLoader5(trainPath, failPath4), std::invalid_argument);

    MnistLoader mnistLoader6(failPath5, testPath);
    CHECK_THROWS_AS(mnistLoader6.load_data_wrapper(), std::invalid_argument);

    printResults();
    return failed;
}
