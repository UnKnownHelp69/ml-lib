#pragma once

#include "types.hpp"
#include <vector>
#include <stdexcept>

class Matrix {
private:
    std::vector<Scalar> dataRow;
    // std::vector<Scalar> data_column; // for now it is not implemented
    int rows, columns;
public:
    Matrix(int _rows, int _columns, std::vector<Scalar> _dataRow) : rows(_rows), columns(_columns), 
        dataRow(std::move(_dataRow)) {
            if (dataRow.size() != rows * columns) 
                throw std::invalid_argument("Size of matrix is not match the real matrix size!");
        }

    int getRows() const { return rows; }
    int getColumns() const { return columns; }
    std::vector<Scalar>& getDataRow() { return dataRow; } 

    Scalar& operator()(int row, int column) {
        if (row < 0 || row >= rows || 
            column < 0 || column >= columns)
                throw std::out_of_range("Index out of range in matrix");
        return dataRow[static_cast<size_t>(row * columns + column)];
    }
    const Scalar& operator()(int row, int column) const {
        if (row < 0 || row >= rows || 
            column < 0 || column >= columns)
                throw std::out_of_range("Index out of range in matrix");
        return dataRow[static_cast<size_t>(row * columns + column)];
    }    

    Matrix operator*(Matrix rightMatrix);
};
