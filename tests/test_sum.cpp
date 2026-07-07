#include <iostream>
#include "sum.hpp"

static int failed = 0;

#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            std::cout << "FAIL [" << __FILE__ << ":" << __LINE__ << "]:" << #expr << '\n'; \
            ++failed; \
        } \
    } while(0) 

int main() {
    CHECK(sum(2, 3) == 5);
    CHECK(sum(2, -3) == -1);
    CHECK(sum(-2, 2) == 0);
    CHECK(sum(-2, 3) == 1);

    if (failed == 0) std::cout << "All passed\n";
    return failed;
}