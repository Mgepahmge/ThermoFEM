add_executable(example-1 example-1.cu)
target_link_libraries(example-1 PRIVATE ThermoFEM::thermofem)

add_executable(example-2 example-2.cu)
target_link_libraries(example-2 PRIVATE ThermoFEM::thermofem)

add_executable(example-3 example-3.cu)
target_link_libraries(example-3 PRIVATE ThermoFEM::thermofem)

add_executable(example-4 example-4.cu)
target_link_libraries(example-4 PRIVATE ThermoFEM::thermofem)

add_executable(solve-matrix solve-matrix.cu)
target_link_libraries(solve-matrix PRIVATE ThermoFEM::thermofem)
if(BUILD_TESTING)
    add_test(NAME MatrixMarketIntegration
        COMMAND solve-matrix "${PROJECT_SOURCE_DIR}/test/matrix-market/poisson.mtx" amg)
endif()
