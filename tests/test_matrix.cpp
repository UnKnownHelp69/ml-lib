#include <vector>
#include <stdexcept>
#include <utility>

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

    // check +, - operators
    Matrix matrix17(3, 3, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    Matrix matrix18(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, -9});
    Matrix matrix19(1, 3, {1, 2, 3});

    Matrix matrix21 = matrix17 - matrix18;
    Matrix matrix22 = matrix17 + matrix18;
    auto rowdata21 = matrix21.getRefDataRow();
    auto rowdata22 = matrix22.getRefDataRow();

    CHECK((rowdata21 == std::vector<Scalar>{0, -1, -2, -3, -4, -5, -6, -7, 10}));
    CHECK((rowdata22 == std::vector<Scalar>{2, 3, 4, 5, 6, 7, 8, 9, -8}));

    CHECK_THROWS_AS(matrix18 - matrix19, std::invalid_argument);
    CHECK_THROWS_AS(matrix18 + matrix19, std::invalid_argument);

    // check sum functions
    CHECK(matrix1.sum() == -2);
    CHECK(matrix1.sumAbs() == 14);
    CHECK(matrix1.sumSq() == 32);

    // check norms (extensible with new norms)
    Matrix matrix23(1, 3, {-3, 4, -12});
    Matrix matrix24(3, 1, {1.5, -0.25, 3.75});
    Matrix matrix25(1, 4, {1, 2, 2, 4});
    Matrix matrix26(1, 2, {0.5, 1.2});
    Matrix matrix27(2, 2, {1, 2, 2, 4});
    Matrix matrix28(2, 2, {0.5, -1.5, 2.0, 0.0});

    CHECK(matrix23.norm("l1") == 19);
    CHECK(matrix24.norm("l1") == 5.5);
    CHECK(matrix25.norm("l2") == 5);
    CHECK(matrix26.norm("l2") == 1.3);
    CHECK(matrix27.norm("frob") == 5);
    CHECK(matrix27.norm() == 5);
    CHECK_THROWS_AS(matrix28.norm("l2"), std::domain_error);
    CHECK_THROWS_AS(matrix28.norm("l1"), std::domain_error);

    // check activation functions (extensible with new activation functions)
    Matrix matrix29(3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    Matrix matrix30(3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    Matrix matrix30_1(3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    matrix29.applyActivation();
    matrix30.applyActivation("sigmoid");
    matrix30_1.applyActivation("sigmoid_prime");
    CHECK((matrix29.getRefDataRow() == std::vector<Scalar>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}));
    CHECK((matrix30.getRefDataRow() == std::vector<Scalar>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}));
    CHECK((matrix30_1.getRefDataRow() == std::vector<Scalar>{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25}));

    CHECK_THROWS_AS(matrix29.applyActivation("blablabla"), std::invalid_argument);

    // check == and != operators
    Matrix matrix31(3, 2, {0, 0, 0, 0, 0, 0});
    Matrix matrix32(3, 2, {0, 0, 0, 0, 0, 0});
    Matrix matrix33(2, 3, {0, 0, 0, 0, 0, 0});
    Matrix matrix34(3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0});

    CHECK(matrix31 == matrix32);
    CHECK((matrix31 == matrix33) == false);
    CHECK((matrix31 == matrix34) == false);
    CHECK((matrix31 != matrix32) == false);
    CHECK(matrix31 != matrix33);
    CHECK(matrix31 != matrix34);

    // check zeros
    Matrix matrix35(1, 3, {-3, 4, -12});
    Matrix matrix36(3, 1, {1.5, -0.25, 3.75});
    matrix35.zeros();
    matrix36.zeros();

    CHECK((matrix35.getRefDataRow() == std::vector<Scalar>{0, 0, 0}));
    CHECK((matrix36.getRefDataRow() == std::vector<Scalar>{0, 0, 0}));

    // check shape
    CHECK((matrix35.shape() == std::pair<size_t, size_t>{1, 3}));
    CHECK((matrix36.shape() == std::pair<size_t, size_t>{3, 1}));

    // check transpose
    Matrix matrix37(2, 2, {1, 2, 3, 4});
    Matrix matrix38(1, 1, std::vector<Scalar>{5});
    Matrix matrix39(4, 2, {1, 2, 3, 4, 5, 6, 7, 8});
    Matrix matrix40(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    matrix37.transpose();
    matrix38.transpose();
    matrix39.transpose();
    matrix40.transpose();

    CHECK((matrix37.getRefDataRow() == std::vector<Scalar>{1, 3, 2, 4}));
    CHECK((matrix38.getRefDataRow() == std::vector<Scalar>{5}));
    CHECK((matrix39.getRefDataRow() == std::vector<Scalar>{1, 3, 5, 7, 2, 4, 6, 8}));
    CHECK((matrix40.getRefDataRow() == std::vector<Scalar>{1, 4, 7, 2, 5, 8, 3, 6, 9}));

    CHECK((matrix37.shape() == std::pair<size_t, size_t>{2, 2}));
    CHECK((matrix38.shape() == std::pair<size_t, size_t>{1, 1}));
    CHECK((matrix39.shape() == std::pair<size_t, size_t>{2, 4}));
    CHECK((matrix40.shape() == std::pair<size_t, size_t>{3, 3}));

    // check argmin, argmax, minimum, maximum
    Matrix matrix41(2, 2, {1, 2, 3, 4});

    CHECK(matrix41.argmax() == 3);
    CHECK(matrix41.argmin() == 0);
    CHECK(matrix41.minimum() == 1);
    CHECK(matrix41.maximum() == 4);

    // check copy constructor
    Matrix matrix42(2, 2, {1, 2, 3, 4});
    Matrix matrix43 = matrix42;

    CHECK(matrix42.getRefDataRow() == matrix43.getRefDataRow());
    CHECK(matrix42.getRows() == matrix43.getRows());
    CHECK(matrix42.getColumns() == matrix43.getColumns());
    matrix42(0, 0) = 0;
    CHECK(matrix42.getRefDataRow() != matrix43.getRefDataRow());

    // check operator =
    Matrix matrix44(2, 2, {0, 0, 0, 0});
    matrix43 = matrix44;
    CHECK(matrix44.getRefDataRow() == matrix43.getRefDataRow());
    CHECK(matrix44.getRows() == matrix43.getRows());
    CHECK(matrix44.getColumns() == matrix43.getColumns());
    matrix44(0, 0) = 1;
    CHECK(matrix44.getRefDataRow() != matrix43.getRefDataRow());

    // check hadamart multiplication
    Matrix matrix45(2, 2, {1, 2, 3, 4});
    Matrix matrix46(2, 2, {1, 2, 3, 4});
    Matrix matrix47(1, 3, {1, 2, 3});
    Matrix matrix48(3, 1, {1, 2, 3});
    Matrix matrix49(3, 1, {1, 2, 3});

    CHECK(((matrix45 % matrix46).getRefDataRow() == std::vector<Scalar>({1, 4, 9, 16})));
    CHECK(((matrix46 % matrix45).getRefDataRow() == std::vector<Scalar>({1, 4, 9, 16})));
    CHECK(((matrix48 % matrix49).getRefDataRow() == std::vector<Scalar>({1, 4, 9})));
    CHECK_THROWS_AS(matrix45 % matrix47, std::invalid_argument);
    CHECK_THROWS_AS(matrix47 % matrix48, std::invalid_argument);

    printResults();
    return failed;
}
