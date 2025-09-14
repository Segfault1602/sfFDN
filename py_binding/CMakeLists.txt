if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    cpp_fdn # Extension name
    # Source files
    cpp_fdn.cpp
)

target_link_libraries(cpp_fdn PRIVATE sfFDN::sfFDN)
target_include_directories(cpp_fdn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(cpp_fdn PRIVATE cxx_std_23)
target_compile_options(cpp_fdn PRIVATE ${SFFDN_SANITIZER})
target_link_options(cpp_fdn PRIVATE ${SFFDN_LINK_OPTIONS})

# cmake-format: off
nanobind_add_stub(
    cpp_fdn_stub
    MODULE cpp_fdn
    OUTPUT cpp_fdn.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:cpp_fdn>
    DEPENDS cpp_fdn
    MARKER_FILE py.typed
)
# cmake-format: on
