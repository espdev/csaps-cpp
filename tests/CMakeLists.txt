
add_executable(test_csaps csaps.cpp)
target_include_directories(test_csaps PUBLIC ${CSAPS_INCLUDE_DIRS})
target_link_libraries(test_csaps csaps)
