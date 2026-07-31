#include "matrix.hpp"

#include <omp.h>

Matrix Matrix::operator*(Matrix rightMatrix) {
    int rowsRight = rightMatrix.getRows();
    int columnsRight = rightMatrix.getColumns();

    if (columns != rowsRight)
        throw std::invalid_argument("Invalid matrix sizes while multiplying");

    std::vector<Scalar> newData;
    newData.resize(static_cast<size_t>(rows * columnsRight));
    Matrix newMatrix(rows, columnsRight, newData);

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) 
    for (int k = 0; k < columns; ++k) 
    for (int j = 0; j < columnsRight; ++j) 
        newMatrix(i, j) += (*this)(i, k) * rightMatrix(k, j);

    return newMatrix;
}
