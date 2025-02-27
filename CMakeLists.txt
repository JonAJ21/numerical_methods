cmake_minimum_required(VERSION 3.10)

project(numerical_methods)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

add_executable(l1-1
    l1-1.cpp
    src/matrix.cpp)

add_executable (l1-2
    l1-2.cpp
    src/matrix.cpp)