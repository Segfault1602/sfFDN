cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 23)

project(sfFDN)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(./fetch_deps.cmake)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(py_binding)
