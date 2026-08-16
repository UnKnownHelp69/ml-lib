cmake_minimum_required(VERSION 3.16)
project(ml-lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found!")
endif()

set(ML_FLAGS
    -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion
    -Wnull-dereference -Wnon-virtual-dtor
    $<$<CONFIG:Debug>:-g -O0 -march=native>
    $<$<CONFIG:Release>:-O3 -march=native>
)

add_library(mllib # add here all tested modules
    src/matrix.cpp
    src/network.cpp
)
target_include_directories(mllib PUBLIC include)
target_compile_options(mllib PRIVATE ${ML_FLAGS})
target_link_libraries(mllib PRIVATE OpenMP::OpenMP_CXX)

add_executable(ml_app src/main.cpp)
target_link_libraries(ml_app PRIVATE mllib)
target_compile_options(ml_app PRIVATE ${ML_FLAGS})
target_link_libraries(ml_app PRIVATE OpenMP::OpenMP_CXX)

enable_testing()
add_subdirectory(tests)
