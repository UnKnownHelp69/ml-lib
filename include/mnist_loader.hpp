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
        trainPath(_trainPath), testPath(_testPath) {}
    std::array<std::vector<std::pair<Matrix, Matrix>>, 3> load_data_wrapper();
};
