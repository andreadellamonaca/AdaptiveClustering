cmake_minimum_required(VERSION 3.14)
project(AdaptiveClustering)

set(CMAKE_CXX_STANDARD 14)

add_executable(AdaptiveClustering main.cpp alglib/dataanalysis.h alglib/dataanalysis.cpp
        alglib/alglibinternal.cpp alglib/alglibinternal.h  alglib/alglibmisc.cpp  alglib/alglibmisc.h
        alglib/ap.cpp  alglib/ap.h  alglib/linalg.cpp  alglib/linalg.h  alglib/optimization.cpp  alglib/optimization.h
        alglib/solvers.cpp  alglib/solvers.h  alglib/specialfunctions.cpp  alglib/specialfunctions.h  alglib/statistics.cpp
        alglib/statistics.h  alglib/stdafx.h)
target_link_libraries(AdaptiveClustering m)