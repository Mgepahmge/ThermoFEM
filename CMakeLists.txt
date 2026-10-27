cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures to build")
endif()

project(ThermoFEM VERSION 0.1.0 DESCRIPTION "CUDA finite element heat conduction solver" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
find_package(CUDAToolkit 12.0 REQUIRED)
option(THERMOFEM_BUILD_EXAMPLES "Build ThermoFEM examples" ON)

add_library(thermofem STATIC src/version.cu src/DataStructure/Memory.cu src/LinearAlgebra/Backend.cu src/LinearAlgebra/IterationContext.cu src/IO/MatrixMarket.cu)
add_library(ThermoFEM::thermofem ALIAS thermofem)
target_include_directories(thermofem PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(thermofem PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(thermofem PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)
set_target_properties(thermofem PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(thermofem_cli src/main.cu)
target_link_libraries(thermofem_cli PRIVATE ThermoFEM::thermofem)
set_target_properties(thermofem_cli PROPERTIES OUTPUT_NAME thermofem)

if(THERMOFEM_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
if(BUILD_TESTING)
    add_subdirectory(test)
endif()

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/build/docs"
        COMMAND Doxygen::doxygen "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Building ThermoFEM documentation" VERBATIM)
endif()

install(TARGETS thermofem thermofem_cli EXPORT ThermoFEMTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.cuh")
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/ThermoFEM)
install(EXPORT ThermoFEMTargets NAMESPACE ThermoFEM:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ThermoFEM)
configure_package_config_file(cmake/ThermoFEMConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ThermoFEMConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ThermoFEM)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ThermoFEMConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ThermoFEMConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ThermoFEMConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ThermoFEM)
