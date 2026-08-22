#pragma once

#include "types.hpp"
#include <vector>
#include <stdexcept>
#include <cassert>
#include <type_traits>

class Matrix {
private:
    size_t rows, columns;
    std::vector<Scalar> dataRow;
    // std::vector<Scalar> data_column; // for now it is not implemented

    static_assert(std::is_pod<Scalar>::value, "Matrix requaries POD type"); // notification for the future
public:
    Matrix(size_t _rows, size_t _columns, std::vector<Scalar> _dataRow) : rows(_rows), columns(_columns), 
        dataRow(std::move(_dataRow)) {
            if (dataRow.size() != rows * columns) 
                throw std::invalid_argument("Size of matrix is not match the real matrix size!");
        }
    Matrix(size_t _rows, size_t _columns, std::vector<std::vector<Scalar>> _dataMatrix) : 
        rows(_rows), columns(_columns), dataRow(_rows * _columns) {
            size_t sz = 0;
            for (auto& row : _dataMatrix) {
                size_t rowSz = row.size();
                sz += rowSz;
                if (rowSz != columns)
                    throw std::invalid_argument("Column size of the matrix is not match the real matrix size!");
            }
            if (sz != rows * columns) 
                throw std::invalid_argument("Row size of the matrix is not match the real matrix size!");
            size_t ind = 0;
            for (auto& row : _dataMatrix) {
                for (auto& x : row) {
                    dataRow[ind++] = x;
                }
            }
        }
    Matrix(size_t _rows, size_t _columns) : rows(_rows), columns(_columns), 
        dataRow(_rows * _columns) {randomize();} // note that it is randoming

    size_t getRows() const { return rows; }
    size_t getColumns() const { return columns; }
    std::vector<Scalar>& getRefDataRow() { return dataRow; } 
    
    // value is int because do not want any suprising attitude 
    Scalar& operator()(int row, int column) {
        if (row < 0 || static_cast<size_t>(row) >= rows || 
            column < 0 || static_cast<size_t>(column) >= columns)
                throw std::out_of_range("Index out of range in matrix");
        return dataRow[static_cast<size_t>(static_cast<size_t>(row) * columns + static_cast<size_t>(column))];
    }
    const Scalar& operator()(int row, int column) const {
        if (row < 0 || static_cast<size_t>(row) >= rows || 
            column < 0 || static_cast<size_t>(column) >= columns)
                throw std::out_of_range("Index out of range in matrix");
        return dataRow[static_cast<size_t>(static_cast<size_t>(row) * columns + static_cast<size_t>(column))];
    }

    Matrix operator-(Matrix& subtractedMatrix);
    Matrix operator+(Matrix& addedMatrix);
    Matrix operator*(Matrix& rightMatrix);

    void randomize();
};
