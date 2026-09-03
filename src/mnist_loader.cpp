#include "mnist_loader.hpp"

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cmath>


std::array<std::vector<std::pair<Matrix, Matrix>>, 3> MnistLoader::load_data_wrapper() {
    /*
    Method wich find by path MNIST, then unpack and return into the 
    3 vectors: training_data, validation_data, test_data in 70, 20, 100
    proportions.

    In the future: add some params.
    */
    std::vector<std::pair<Matrix, Matrix>> tmpForDistrData, training_data, validation_data, test_data;
    std::ifstream trainFile(trainPath), testFile(testPath);
    std::string line;
    test_data = {};

    if (!trainFile.is_open()) // train file have to exists
        throw std::invalid_argument("The chosen train file does not exist: " + trainPath + '\n');
    if (testPath != "" && !testFile.is_open()) // test file may not exists
        throw std::invalid_argument("The chosen test file does not exist: " + testPath + '\n');

    while (std::getline(trainFile, line)) {
        //std::cout << line << "\n";
        std::vector<Scalar> dataRow;
        bool fndAns = false; int ans = 0;
        std::string nowNumString = "";

        for (auto& c : line) {
            if (c == ',') {
                if (nowNumString == "")
                    throw std::runtime_error("Smth got wrong while parsing csv trainFile (nowNumString is empty).\n");
                if (!fndAns) {
                    ans = std::stoi(nowNumString);
                    fndAns = true;
                } else {
                    int tmp = std::stoi(nowNumString);
                    dataRow.push_back(static_cast<Scalar>(tmp) / 255.0);
                }
                nowNumString = "";
                continue;
            }
            nowNumString += c;
        }

        // if string is not finished on comma (as in MNIST exactly)
        if (nowNumString != "") {
            if (!fndAns) {
                ans = std::stoi(nowNumString);
                fndAns = true;
            } else {
                int tmp = std::stoi(nowNumString);
                dataRow.push_back(static_cast<Scalar>(tmp) / 255.0);
            }
        }

        //std::cout << dataRow.size() << " " << ans << "\n\n\n";
        //for (auto& c : dataRow) std::cout << c << ","; std::cout << "\n";

        if (dataRow.size() != 784)
            throw std::runtime_error("Mnist row train input is not 784 numbers.\n");

        Matrix y(10, 1); y.zeros(); y(ans, 0) = 1;
        Matrix x(784, 1, dataRow);
        tmpForDistrData.push_back({x, y});
    }

    //std::cout << "GOING TO SPLIT\n";

    size_t allLen = tmpForDistrData.size();
    size_t trainingLen = static_cast<size_t>(std::ceil(static_cast<double>(allLen) * 0.8));
    size_t valLen = allLen - trainingLen;
    training_data.reserve(trainingLen); validation_data.reserve(valLen);
    for (size_t i = 0; i < trainingLen; ++i)
        training_data.emplace_back(std::move(tmpForDistrData[i]));
    for (size_t i = 0; i < valLen; ++i)
        validation_data.emplace_back(std::move(tmpForDistrData[i + trainingLen]));

    if (testPath == "") return {training_data, validation_data, test_data};

    while (std::getline(testFile, line)) {
        std::vector<Scalar> dataRow;
        bool fndAns = false; int ans = 0;
        std::string nowNumString = "";

        for (auto& c : line) {
            if (c == ',') {
                if (nowNumString == "")
                    throw std::runtime_error("Smth got wrong while parsing csv testFile (nowNumString is empty).\n");
                if (!fndAns) {
                    ans = std::stoi(nowNumString);
                    fndAns = true;
                } else {
                    int tmp = std::stoi(nowNumString);
                    dataRow.push_back(static_cast<Scalar>(tmp) / 255.0);
                }
                nowNumString = "";
                continue;
            }
            nowNumString += c;
        }

        // if string is not finished on comma (as in MNIST exactly)
        if (nowNumString != "") {
            if (!fndAns) {
                ans = std::stoi(nowNumString);
                fndAns = true;
            } else {
                int tmp = std::stoi(nowNumString);
                dataRow.push_back(static_cast<Scalar>(tmp) / 255.0);
            }
        }

        if (dataRow.size() != 784)
            throw std::runtime_error("Mnist row test input is not 784 numbers.\n");

        Matrix y(10, 1); y.zeros(); y(ans, 0) = 1;
        Matrix x(784, 1, dataRow);
        test_data.push_back({x, y});
        // break; // to test only one run uncomment
    }

    return {training_data, validation_data, test_data};
}
