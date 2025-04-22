include_guard(GLOBAL)

if(APPLE)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm@22/bin/clang++")
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm@22/bin/clang")
else()
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_C_COMPILER "clang")
endif()
