add_executable(test_sum test_sum.cpp)
target_link_libraries(test_sum PRIVATE mllib)
add_test(NAME test_sum COMMAND test_sum)
