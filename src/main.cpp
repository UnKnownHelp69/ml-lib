#include <iostream>
#include <chrono>

#include "network.hpp"
#include "mnist_loader.hpp"

int main() {
    auto t0 = std::chrono::steady_clock::now(); // programm started

    Network net({784, 30, 10});
    std::string trainPath = "../MNIST/mnist_train.csv";
    std::string testPath = "../MNIST/mnist_test.csv";
    MnistLoader mnistLoader(trainPath, testPath);
    auto [training_data, validation_data, test_data] = mnistLoader.load_data_wrapper();

    auto t1 = std::chrono::steady_clock::now(); // learning started, reading finished

    std::cout << training_data.size() << " " << validation_data.size() << " " << test_data.size() << "\n";
    std::cout << "Education started, files was reading:\n";
    std::cout << "  " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    auto t2 = std::chrono::steady_clock::now(); // education started
    net.SGD(training_data, 30, 10, 3.0, test_data);\
    auto t3 = std::chrono::steady_clock::now(); // education finished
    std::cout << "Education finished for:\n";
    std::cout << "  " << std::chrono::duration<double, std::milli>(t3 - t2).count() / 1000.0 << " s\n";

    return 0;
}
