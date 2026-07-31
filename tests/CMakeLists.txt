add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE mllib)
add_test(NAME test_matrix COMMAND test_matrix)
