cmake_minimum_required(VERSION 3.14)
project(AdaptiveClustering)

set(CMAKE_CXX_STANDARD 14)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIR})

add_executable(AdaptiveClustering main.cpp adaptive_clustering.cpp adaptive_clustering.h error.cpp error.h)
target_link_libraries(AdaptiveClustering m ${ARMADILLO_LIBRARIES})