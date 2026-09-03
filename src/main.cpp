#include <iostream>

#include "Network.hpp"
#include "mnist_loader.hpp"

int main() {
    Network net({784, 30, 10});
    std::string trainPath = "../MNIST/mnist_train.csv";
    std::string testPath = "../MNIST/mnist_test.csv";
    MnistLoader mnistLoader(trainPath, testPath);
    auto [training_data, validation_data, test_data] = mnistLoader.load_data_wrapper();
    std::cout << training_data.size() << " " << validation_data.size() << " " << test_data.size() << "\n";
    std::cout << "Education started\n";
    net.SGD(training_data, 30, 10, 3.0, test_data);
    std::cout << "Education finished\n";

    return 0;
}
