cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(csaps
    VERSION 0.1.0
)

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

# Generate and install CMake config, version and target files.
include(CMakePackageConfigHelpers)
set(CONFIG_INSTALL_DIR "lib/cmake/${PROJECT_NAME}")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(VERSION_CONFIG "${GENERATED_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(PROJECT_CONFIG "${GENERATED_DIR}/${PROJECT_NAME}Config.cmake")
write_basic_package_version_file(
    "${VERSION_CONFIG}" VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "cmake/Config.cmake.in"
    "${PROJECT_CONFIG}"
    INSTALL_DESTINATION "${CONFIG_INSTALL_DIR}"
)

install(FILES "${PROJECT_CONFIG}" "${VERSION_CONFIG}"
    DESTINATION "${CONFIG_INSTALL_DIR}")
install(EXPORT csaps::csaps
  NAMESPACE csaps::
  FILE csapsTargets.cmake
  DESTINATION "${CONFIG_INSTALL_DIR}")
