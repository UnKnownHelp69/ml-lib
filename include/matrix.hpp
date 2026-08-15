#pragma once

#include "types.hpp"
#include <vector>
#include <stdexcept>
#include <cassert>
#include <type_traits>

class Matrix {
private:
    std::vector<Scalar> dataRow;
    // std::vector<Scalar> data_column; // for now it is not implemented
    int rows, columns;

    static_assert(std::is_pod<Scalar>::value, "Matrix requaries POD type"); // notification for the future
public:
    Matrix(int _rows, int _columns, std::vector<Scalar> _dataRow) : rows(_rows), columns(_columns), 
        dataRow(std::move(_dataRow)) {
            if (dataRow.size() != rows * columns) 
                throw std::invalid_argument("Size of matrix is not match the real matrix size!");
        }
    Matrix(int _rows, int _columns, std::vector<std::vector<Scalar>> _dataMatrix) : rows(_rows), columns(_columns), 
        dataRow(_rows * _columns) {
            int sz = 0;
            for (auto& row : _dataMatrix) {
                int rowSz = row.size();
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
    Matrix(int _rows, int _columns) : rows(_rows), columns(_columns), dataRow(_rows * _columns) 
            {randomize();}

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

    void randomize();
};
