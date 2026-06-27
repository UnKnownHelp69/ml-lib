cmake_minimum_required(VERSION 3.16)
project(ml-lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ML_FLAGS
    -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion
    -Wnull-dereference -Wnon-virtual-dtor
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -march=native>
)

add_library(mllib 
    src/sum.cpp
)
target_include_directories(mllib PUBLIC include)
target_compile_options(mllib PRIVATE ${ML_FLAGS})

add_executable(ml_app src/main.cpp)
target_link_libraries(ml_app PRIVATE mllib)
target_compile_options(ml_app PRIVATE ${ML_FLAGS})

enable_testing()
add_subdirectory(tests)
