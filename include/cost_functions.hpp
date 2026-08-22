#pragma once

#include <stdexcept>

#include "types.hpp"
#include "matrix.hpp"
#include "cmath"


inline Scalar MSE(const Matrix& w, const Matrix& b) {
    size_t wRows = w.getRows(), wColumns = w.getColumns();
    if (wRows != b.getRows() || wColumns != b.getColumns())
        throw std::invalid_argument("Invalid matrix sizes while MSE evaluation");

    const auto& dataW = w.getRefDataRow();
    const auto& dataB = b.getRefDataRow();

    size_t totalSize = wRows * wColumns;
    Scalar sumSq = 0;
    // for exclude data race making local copy for every thread
    #pragma omp parallel for reduction(+:sumSq) // may be later if(totalSize > trashhold)
    for (size_t i = 0; i < totalSize; ++i) {
        Scalar diff = dataW[i] - dataB[i];
        sumSq += diff * diff; 
    }

    return sumSq / static_cast<Scalar>(totalSize);
}
