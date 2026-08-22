add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE mllib)
add_test(NAME test_matrix COMMAND test_matrix)

add_executable(test_network test_network.cpp)
target_link_libraries(test_network PRIVATE mllib)
add_test(NAME test_network COMMAND test_network)

add_executable(test_cost_functions test_cost_functions.cpp)
target_link_libraries(test_cost_functions PRIVATE mllib)
add_test(NAME test_cost_functions COMMAND test_cost_functions)
