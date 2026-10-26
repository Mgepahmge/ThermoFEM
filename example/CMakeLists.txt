add_executable(example-1 example-1.cu)
target_link_libraries(example-1 PRIVATE ThermoFEM::thermofem)

add_executable(example-2 example-2.cu)
target_link_libraries(example-2 PRIVATE ThermoFEM::thermofem)
