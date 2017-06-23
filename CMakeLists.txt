cmake_minimum_required(VERSION 3.2)

project(csaps)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

enable_testing()

set(BUILD_TESTING OFF CACHE BOOL "Build tests")

find_package(Eigen3 3.1 REQUIRED)

set(csaps_SOURCES
    csaps.h
    csaps.cpp
)

add_library(csaps STATIC ${csaps_SOURCES})
target_include_directories(csaps PUBLIC ${EIGEN3_INCLUDE_DIR})

if (BUILD_TESTING)
    add_executable(test_csaps test_csaps.cpp)
    target_include_directories(test_csaps PRIVATE ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(test_csaps csaps)

    add_test(test_csaps test_csaps)
endif()
