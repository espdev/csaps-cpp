cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(csaps)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(BUILD_TESTING OFF CACHE BOOL "Build tests")

find_package(Eigen3 3.1 REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

enable_testing()

if (BUILD_TESTING)
    add_executable(test_csaps test_csaps.cpp)
    target_include_directories(test_csaps PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_csaps csaps)

    add_test(test_csaps test_csaps)
endif()
