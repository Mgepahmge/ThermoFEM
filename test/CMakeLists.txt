find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared CRT" FORCE)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(thermofem_tests version_test.cpp data_structure_test.cpp linear_algebra_test.cpp)
target_link_libraries(thermofem_tests PRIVATE ThermoFEM::thermofem GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(thermofem_tests NO_PRETTY_VALUES)
