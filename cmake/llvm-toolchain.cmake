include_guard(GLOBAL)

if(APPLE)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang++")
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang")
else()
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_C_COMPILER "clang")
endif()
