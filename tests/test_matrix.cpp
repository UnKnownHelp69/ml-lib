#include <vector>

#include "matrix.hpp"
#include "types.hpp"
#include "test_utlis.hpp"

int main() {
    // Linear vector int
    Matrix matrix1(3, 3, {0, 1, 2, 3, -1, -2, -3, -2, 0});
    Matrix matrix2(3, 2, {1, 1, 1, 1, 1, 1});
    Matrix matrix3(1, 1, std::vector<Scalar>{0});

    CHECK(matrix1(0, 0) == 0);
    CHECK(matrix1(0, 1) == 1);
    CHECK(matrix1(0, 2) == 2);
    CHECK(matrix1(1, 0) == 3);
    CHECK(matrix1(1, 1) == -1);
    CHECK(matrix1(1, 2) == -2);
    CHECK(matrix1(2, 0) == -3);
    CHECK(matrix1(2, 1) == -2);
    CHECK(matrix1(2, 2) == 0);

    Matrix matrix4 = (matrix1 * matrix2);

    CHECK_THROWS_AS(matrix4(1000, 1000) == 5, std::out_of_range);
    CHECK_THROWS_AS(matrix1 * matrix3, std::invalid_argument);
    
    CHECK(matrix2(2, 1) == 1);
    
    CHECK(matrix4(0, 0) == 3);
    CHECK(matrix4(0, 1) == 3);
    CHECK(matrix4(1, 0) == 0);
    CHECK(matrix4(1, 1) == 0);
    CHECK(matrix4(2, 0) == -5);

    CHECK(matrix1.getRows() == 3);
    CHECK(matrix1.getColumns() == 3);
    CHECK(matrix2.getRows() == 3);
    CHECK(matrix2.getColumns() == 2);
    CHECK(matrix4.getRows() == 3);
    CHECK(matrix4.getColumns() == 2);

    // 2D vector init
    Matrix matrix5(3, 3, {{0, 1, 2}, {3, -1, -2}, {-3, -2, 0}});
    Matrix matrix6(3, 2, {{1, 1}, {1, 1}, {1, 1}});
    Matrix matrix7(1, 1, std::vector<Scalar>{0});

    CHECK(matrix5(0, 0) == 0);
    CHECK(matrix5(0, 1) == 1);
    CHECK(matrix5(0, 2) == 2);
    CHECK(matrix5(1, 0) == 3);
    CHECK(matrix5(1, 1) == -1);
    CHECK(matrix5(1, 2) == -2);
    CHECK(matrix5(2, 0) == -3);
    CHECK(matrix5(2, 1) == -2);
    CHECK(matrix5(2, 2) == 0);

    CHECK_THROWS_AS(Matrix matrix8(1, 1, {1, 2, 3}), std::invalid_argument);
    CHECK_THROWS_AS(Matrix matrix9(2, 3, {1, 2, 3}), std::invalid_argument);
    CHECK_THROWS_AS(Matrix matrix10(3, 2, {{1, 2, 3}, {4, 5, 6}}), std::invalid_argument);
    CHECK_THROWS_AS(Matrix matrix11(2, 3, {{1, 2}, {3, 4}, {5, 6}}), std::invalid_argument);
    CHECK_THROWS_AS(Matrix matrix12(3, 2, {{1, 2}, {3, 4}, {5, 6}, {7, 8}}), std::invalid_argument);
    CHECK_THROWS_AS(Matrix matrix13(3, 2, {{1, 2}, {3, 4}}), std::invalid_argument);
    
    CHECK(matrix6(2, 1) == 1);

    CHECK(matrix5.getRows() == 3);
    CHECK(matrix5.getColumns() == 3);
    CHECK(matrix6.getRows() == 3);
    CHECK(matrix6.getColumns() == 2);
    CHECK(matrix7.getRows() == 1);
    CHECK(matrix7.getColumns() == 1);

    // empty init, for now do not check random so just check if nothing breaks
    Matrix matrix14(3, 3);
    Matrix matrix15(3, 2);
    Matrix matrix16(1, 1);

    printResults();
    return failed;
}
