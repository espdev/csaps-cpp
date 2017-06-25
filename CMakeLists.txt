cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(csaps)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(BUILD_TESTING OFF CACHE BOOL "Build tests")

find_package(Eigen3 3.1 REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

set(CSAPS_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/thirdparty
)

if (BUILD_TESTING)
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
endif()
