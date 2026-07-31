#include <iostream>
#include "matrix.hpp"

static int failed = 0;
static int passed = 0;


#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            std::cout << "FAIL [" << __FILE__ << ":" << __LINE__ << "]:" << #expr << std::endl; \
            ++failed; \
        } else {\
            std::cout << "Pass [" << __FILE__ << ":" << __LINE__ << "]:" << #expr << std::endl; \
            ++passed; \
        } \
    } while(0)


#define CHECK_THROWS_AS(expr, expected_error) \
    do { \
        bool caught = false; \
        try { \
            expr; \
        } catch (const expected_error& e) { \
            ++passed; \
            caught = true; \
            std::cout << "PASS [" << __FILE__ << ":" << __LINE__ << "]:" <<  #expr << " threw " \
                << #expected_error << ":" << e.what() << std::endl; \
        } catch (const std::exception& e) { \
            ++failed; \
            caught = true; \
            std::cout << "FAIL [" << __FILE__ << ":" << __LINE__ << "]:" << #expr << " threw wrong error type. Got:" \
                << typeid(e).name() << ", expected: " << #expected_error << std::endl; \
        } catch (...) { \
            ++failed; \
            caught = true; \
            std::cout << "FAIL [" << __FILE__ << ":" << __LINE__ << "]:" << #expr \
                << " threw unknown error type, expected: " << #expected_error << std::endl; \
        } \
        if (!caught) { \
            ++failed; \
            std::cout << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: did not threw an error: " \
                << #expr << std::endl; \
        } \
    } while(0)

int main() {
    Matrix matrix1(3, 3, {0, 1, 2, 3, -1, -2, -3, -2, 0});
    Matrix matrix2(3, 2, {1, 1, 1, 1, 1, 1});
    Matrix matrix3(1, 1, {0});

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

    if (failed == 0) std::cout << "All passed\n";
    return failed;
}