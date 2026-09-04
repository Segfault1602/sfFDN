include_guard(GLOBAL)

if(APPLE)
    # Homebrew keg names change with every LLVM release, so discover the newest usable prefix
    # instead of hardcoding one. Override with -DCMAKE_CXX_COMPILER= to pin a specific toolchain.
    if(NOT DEFINED CMAKE_CXX_COMPILER OR NOT DEFINED CMAKE_C_COMPILER)
        # The unversioned prefix tracks the current default and is the most stable; versioned kegs come and go with
        # upgrades, so they are only a fallback for a machine that has no default llvm installed.
        set(_sffdn_llvm_prefixes "/opt/homebrew/opt/llvm" "/usr/local/opt/llvm")
        file(GLOB _sffdn_llvm_kegs "/opt/homebrew/opt/llvm@*" "/usr/local/opt/llvm@*")
        list(SORT _sffdn_llvm_kegs COMPARE NATURAL ORDER DESCENDING)
        list(APPEND _sffdn_llvm_prefixes ${_sffdn_llvm_kegs})

        foreach(_sffdn_prefix IN LISTS _sffdn_llvm_prefixes)
            if(EXISTS "${_sffdn_prefix}/bin/clang++" AND EXISTS "${_sffdn_prefix}/bin/clang")
                set(_sffdn_llvm_root "${_sffdn_prefix}")
                break()
            endif()
        endforeach()

        if(NOT DEFINED _sffdn_llvm_root)
            message(
                FATAL_ERROR
                "No Homebrew LLVM toolchain found under /opt/homebrew/opt or /usr/local/opt. "
                "Install one with 'brew install llvm', or pass -DCMAKE_CXX_COMPILER= explicitly."
            )
        endif()

        if(NOT DEFINED CMAKE_CXX_COMPILER)
            set(CMAKE_CXX_COMPILER "${_sffdn_llvm_root}/bin/clang++")
        endif()
        if(NOT DEFINED CMAKE_C_COMPILER)
            set(CMAKE_C_COMPILER "${_sffdn_llvm_root}/bin/clang")
        endif()
    endif()
else()
    if(NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "clang++")
    endif()
    if(NOT DEFINED CMAKE_C_COMPILER)
        set(CMAKE_C_COMPILER "clang")
    endif()
endif()
