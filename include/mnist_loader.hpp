#pragma once

#include <array>
#include <vector>
#include <utility>
#include <string>

#include "matrix.hpp"

class MnistLoader{
private:
    std::string trainPath, testPath;
public:
    MnistLoader(std::string _trainPath, std::string _testPath = "") : 
        trainPath(_trainPath), testPath(_testPath) {
            if (trainPath == "")
                throw std::invalid_argument("The train path can not be empty.");
            if (testPath == "")
                throw std::invalid_argument("The test path can not be empty."); 
        }
    std::array<std::vector<std::pair<Matrix, Matrix>>, 3> load_data_wrapper();
};
