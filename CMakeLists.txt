cmake_minimum_required(VERSION 3.14)
project(AdaptiveClustering)

set(CMAKE_CXX_STANDARD 14)

add_executable(AdaptiveClustering main.cpp
        alglib/alglibinternal.h alglib/alglibinternal.cpp alglib/alglibmisc.h alglib/alglibmisc.cpp
        alglib/ap.h alglib/ap.cpp alglib/dataanalysis.h alglib/dataanalysis.cpp alglib/linalg.h alglib/linalg.cpp
        alglib/optimization.h alglib/optimization.cpp alglib/solvers.h alglib/solvers.cpp alglib/specialfunctions.h
        alglib/specialfunctions.cpp alglib/statistics.h alglib/statistics.cpp)
target_link_libraries(AdaptiveClustering m)