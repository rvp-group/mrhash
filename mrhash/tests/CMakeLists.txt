enable_testing()

add_executable(test_algebra_operations test_algebra_operations.cu)
target_link_libraries(test_algebra_operations PRIVATE GTest::gtest_main ${GEOM_LIB_NAME})
target_include_directories(test_algebra_operations PRIVATE ${PROJECT_SOURCE_DIR}/src/)

add_executable(test_cuda_matrix test_cuda_matrix.cu)
target_link_libraries(test_cuda_matrix PRIVATE GTest::gtest_main ${GEOM_LIB_NAME})
target_include_directories(test_cuda_matrix PRIVATE ${PROJECT_SOURCE_DIR}/src/)

add_executable(test_projections test_projections.cu)
target_link_libraries(test_projections PRIVATE GTest::gtest_main ${GEOM_LIB_NAME})
target_include_directories(test_projections PRIVATE ${PROJECT_SOURCE_DIR}/src/)

add_executable(test_hash_utils test_hash_utils.cu)
target_link_libraries(test_hash_utils GTest::gtest_main ${GEOM_LIB_NAME})
target_include_directories(test_hash_utils PRIVATE ${PROJECT_SOURCE_DIR}/src/)

add_executable(test_streamer test_streamer.cu)
target_link_libraries(test_streamer GTest::gtest_main ${GEOM_LIB_NAME})
target_include_directories(test_streamer PRIVATE ${PROJECT_SOURCE_DIR}/src/)

add_executable(test_grid_serializer test_grid_serializer.cu)
target_link_libraries(test_grid_serializer GTest::gtest_main ${GEOM_LIB_NAME})
target_include_directories(test_grid_serializer PRIVATE ${PROJECT_SOURCE_DIR}/src/)

add_executable(test_marching_cubes test_marching_cubes.cpp)
target_link_libraries(test_marching_cubes GTest::gtest_main ${GEOM_LIB_NAME})
target_include_directories(test_marching_cubes PRIVATE ${PROJECT_SOURCE_DIR}/src/)

include(GoogleTest)
gtest_discover_tests(test_algebra_operations)
gtest_discover_tests(test_cuda_matrix)
gtest_discover_tests(test_projections)
gtest_discover_tests(test_hash_utils)
gtest_discover_tests(test_streamer)
gtest_discover_tests(test_grid_serializer)
gtest_discover_tests(test_marching_cubes)

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/  # Change to the build directory
    DEPENDS test_algebra_operations test_cuda_matrix test_projections
            test_hash_utils test_streamer test_grid_serializer test_marching_cubes
)
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
