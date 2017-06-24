cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(csaps)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(BUILD_TESTING OFF CACHE BOOL "Build tests")

find_package(Eigen3 3.1 REQUIRED)

set(csaps_SOURCES
    csaps.h
    csaps.cpp
)

add_library(csaps STATIC ${csaps_SOURCES})
target_include_directories(csaps PUBLIC ${EIGEN3_INCLUDE_DIR})
target_compile_features(csaps PUBLIC cxx_std_11)

enable_testing()

if (BUILD_TESTING)
    add_executable(test_csaps test_csaps.cpp)
    target_include_directories(test_csaps PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(test_csaps csaps)

    add_test(test_csaps test_csaps)
endif()
