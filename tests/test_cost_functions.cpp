#include "test_utlis.hpp"
#include "cost_functions.hpp"
#include "matrix.hpp"


int main() {
    Matrix matrix1(1, 3, {1, 2, 3});
    Matrix matrix2(1, 3, {1, 2, 0});
    Matrix matrix3(3, 1, {1.0, -2.0, 0});
    Matrix matrix4(3, 1, {0.0, -4.0, 4.0});
    Matrix matrix5(2, 2, {0, 0, 0, 0});
    Matrix matrix6(2, 2, {0, 0, 0, 6});
    Matrix matrix7(2, 2, {0, 0, 0, 0});

    CHECK(MSE(matrix1, matrix2) == 3);
    CHECK(MSE(matrix3, matrix4) == 7);
    CHECK(MSE(matrix5, matrix6) == 9);
    CHECK(MSE(matrix5, matrix7) == 0);

    CHECK_THROWS_AS(MSE(matrix1, matrix3), std::invalid_argument);
    CHECK_THROWS_AS(MSE(matrix1, matrix5), std::invalid_argument);
    CHECK_THROWS_AS(MSE(matrix3, matrix5), std::invalid_argument);

    printResults();
    return failed;
}