nanobind_add_module(pygeowrapper MODULE ../geowrapper.cpp pygeowrapper.cpp)
target_link_libraries(pygeowrapper PRIVATE
    ${GEOM_LIB_NAME}
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
nanobind_add_stub(
    pygeowrapper_stub
    MODULE pygeowrapper
    OUTPUT pygeowrapper.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:pygeowrapper>
    DEPENDS pygeowrapper
)
endif()

install(TARGETS pygeowrapper DESTINATION ..)
