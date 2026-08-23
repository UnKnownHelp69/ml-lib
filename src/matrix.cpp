#include "matrix.hpp"
#include "activation_functions.hpp"

#include <algorithm>
#include <random>
#include <omp.h>


Matrix Matrix::operator-(Matrix& addedMatrix) {
    size_t rowsAdded = addedMatrix.getRows();
    size_t columnsAdded = addedMatrix.getColumns();

    if (columns != columnsAdded || rows != rowsAdded)
        throw std::invalid_argument("Invalid matrix sizes while subtracking");

    std::vector<Scalar> newData(rows * columns);    
    auto refAddedMatrix = addedMatrix.getRefDataRow();

    #pragma omp parallel for // may be later add if(rows * columns > trashHoldForHardware)
    for (size_t i = 0; i < rows * columns; ++i) 
        newData[i] = dataRow[i] - refAddedMatrix[i];

    Matrix newMatrix(rows, columns, newData);
    return newMatrix;
}

Matrix Matrix::operator+(Matrix& addedMatrix) {
    size_t rowsAdded = addedMatrix.getRows();
    size_t columnsAdded = addedMatrix.getColumns();

    if (columns != columnsAdded || rows != rowsAdded)
        throw std::invalid_argument("Invalid matrix sizes while adding");

    std::vector<Scalar> newData(rows * columns);    
    auto refAddedMatrix = addedMatrix.getRefDataRow();

    #pragma omp parallel for // may be later add if(totalSize > trashHolfForHardware)
    for (size_t i = 0; i < rows * columns; ++i) 
        newData[i] = dataRow[i] + refAddedMatrix[i];

    Matrix newMatrix(rows, columns, newData);
    return newMatrix;
}

Matrix Matrix::operator*(Matrix& rightMatrix) {
    size_t rowsRight = rightMatrix.getRows();
    size_t columnsRight = rightMatrix.getColumns();

    if (columns != rowsRight)
        throw std::invalid_argument("Invalid matrix sizes while multiplying");

    std::vector<Scalar> newData;
    newData.resize(rows * columnsRight);
    Matrix newMatrix(rows, columnsRight, newData);

    #pragma omp parallel for
    for (int i = 0; static_cast<size_t>(i) < rows; ++i) 
    for (int k = 0; static_cast<size_t>(k) < columns; ++k) 
    for (int j = 0; static_cast<size_t>(j) < columnsRight; ++j) 
        newMatrix(i, j) += (*this)(i, k) * rightMatrix(k, j);

    return newMatrix;
}

bool Matrix::operator==(Matrix& rightMatrix) {
    return (columns == rightMatrix.getColumns()) && (rows == rightMatrix.getRows())
            && (dataRow == rightMatrix.getRefDataRow());
}

bool Matrix::operator!=(Matrix& rightMatrix) {
    return (columns != rightMatrix.getColumns()) || (rows != rightMatrix.getRows())
            || (dataRow != rightMatrix.getRefDataRow());
}

void Matrix::transpose() {
    std::vector<Scalar> newData(rows * columns);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            newData[j * rows + i] = dataRow[i * columns + j];
        }
    }
    dataRow = std::move(newData);

    std::swap(columns, rows);
}

void Matrix::randomize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Scalar> dist(0.0, 1.0);

    for (size_t i = 0; i < rows * columns; ++i) {
        dataRow[i] = dist(gen);
    }
}

void Matrix::zeros() {
    for (size_t i = 0; i < rows * columns; ++i)
        dataRow[i] = 0;
}

Scalar Matrix::norm(std::string_view norm) const {
    if (norm == "frob"){
        return std::sqrt(sumSq());
    } else if (norm == "l1") {
        if (rows != 1 && columns != 1) 
            throw std::domain_error("L1-norm is only for vectors for now");
        return sumAbs();
    } else if (norm == "l2") {
        if (rows != 1 && columns != 1) 
            throw std::domain_error("L2-norm is only for vectors for now");
        return std::sqrt(sumSq());
    } else {
        throw std::invalid_argument("Invalid norm name");
    }
}

void Matrix::applyActivation(std::string_view activationFunctionName) {
    Scalar (*activationFunction)(Scalar) = nullptr;
    if (activationFunctionName == "sigmoid") {
        activationFunction = &sigmoid;
    } else {
        throw std::invalid_argument("Invalid name of the activation function.");
    }
    for (size_t i = 0; i < rows * columns; ++i)
        dataRow[i] = activationFunction(dataRow[i]);
}

size_t Matrix::argmax() {
    if (dataRow.size() == 0) 
        throw std::domain_error("Size of matrix is zero, so can not find max el.");

    size_t ind = 0;
    Scalar maxEl = dataRow[0];

    for (size_t i = 0; i < rows * columns; ++i) {
        Scalar now = dataRow[i];
        if (now > maxEl) {
            maxEl = now;
            ind = i;
        }
    }
    return ind;
}

size_t Matrix::argmin() {
    if (dataRow.size() == 0) 
        throw std::domain_error("Size of matrix is zero, so can not find max el.");

    size_t ind = 0;
    Scalar minEl = dataRow[0];

    for (size_t i = 0; i < rows * columns; ++i) {
        Scalar now = dataRow[i];
        if (now < minEl) {
            minEl = now;
            ind = i;
        }
    }
    return ind;
}

Scalar Matrix::maximum() {
    if (dataRow.size() == 0) 
        throw std::domain_error("Size of matrix is zero, so can not find max el.");

    Scalar maxEl = dataRow[0];
    for (size_t i = 0; i < rows * columns; ++i)
        maxEl = std::max(maxEl, dataRow[i]);
    return maxEl;
}

Scalar Matrix::minimum() {
    if (dataRow.size() == 0) 
        throw std::domain_error("Size of matrix is zero, so can not find max el.");

    Scalar minEl = dataRow[0];
    for (size_t i = 0; i < rows * columns; ++i) 
        minEl = std::min(minEl, dataRow[i]);
    return minEl;
}
