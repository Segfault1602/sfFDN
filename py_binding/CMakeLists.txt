if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()

find_package(
    Python 3.12
    COMPONENTS Interpreter ${DEV_MODULE}
    REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    cpp_fdn # Extension name
    # Source files
    cpp_fdn.cpp
    )
target_link_libraries(cpp_fdn PRIVATE fdn)
