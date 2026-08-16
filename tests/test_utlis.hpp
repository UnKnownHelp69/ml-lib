#pragma once
#include <iostream>

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

void printResults() {
    if (!failed) {
        std::cout << "\nALL PASSED\n";
        return;
    }
    std::cout << "\nSOME FAILURES:\n    Failed: " << failed << "; Passed: " << passed 
                << "; Total: " << failed + passed << "\n";
}
