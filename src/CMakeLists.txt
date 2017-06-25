set(_LIB_NAME csaps)

set(_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/csaps.h
    ${CMAKE_CURRENT_SOURCE_DIR}/csaps.cpp
)

add_library(${_LIB_NAME} STATIC ${_SOURCES})
target_include_directories(${_LIB_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
target_compile_features(${_LIB_NAME} PUBLIC cxx_std_11)
