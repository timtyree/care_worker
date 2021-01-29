# cmake_minimum_required(VERSION <specify CMake version here>)
project(care_worker)

set(CMAKE_CXX_STANDARD 14)

add_library(care_worker library.cpp library.h)
