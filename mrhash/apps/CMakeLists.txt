add_executable(streamer_example streamer_example.cu)
target_include_directories(streamer_example
  PUBLIC  ${PROJECT_SOURCE_DIR}/src/)
target_link_libraries(streamer_example
                      ${GEOM_LIB_NAME}
                      yaml-cpp)
