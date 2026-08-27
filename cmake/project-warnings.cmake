if(MSVC)
    set(SFFDN_WARNINGS_CXX /W3 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(SFFDN_WARNINGS_CXX
        -Wall
        -Wextra
        -Wpedantic
        -Wno-sign-compare
        -Wno-language-extension-token
        -Wno-c2y-extensions
        -Wunsafe-buffer-usage
    )
endif()

add_library(sfFDN_warnings INTERFACE)
add_library(sfFDN::sfFDN_warnings ALIAS sfFDN_warnings)
target_compile_options(sfFDN_warnings INTERFACE ${SFFDN_WARNINGS_CXX})

if(SFFDN_ENABLE_FUNCTION_EFFECTS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        message(FATAL_ERROR "SFFDN_ENABLE_FUNCTION_EFFECTS requires Clang 19 or newer")
    endif()

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
        message(
            FATAL_ERROR
            "SFFDN_ENABLE_FUNCTION_EFFECTS requires Clang 19 or newer; found ${CMAKE_CXX_COMPILER_VERSION}"
        )
    endif()

    include(CheckCXXCompilerFlag)
    unset(SFFDN_HAS_WFUNCTION_EFFECTS CACHE)
    unset(SFFDN_HAS_WFUNCTION_EFFECT_REDECLARATIONS CACHE)
    unset(SFFDN_HAS_WPERF_CONSTRAINT_IMPLIES_NOEXCEPT CACHE)
    check_cxx_compiler_flag("-Wfunction-effects" SFFDN_HAS_WFUNCTION_EFFECTS)
    check_cxx_compiler_flag(
        "-Wfunction-effect-redeclarations"
        SFFDN_HAS_WFUNCTION_EFFECT_REDECLARATIONS
    )
    check_cxx_compiler_flag(
        "-Wperf-constraint-implies-noexcept"
        SFFDN_HAS_WPERF_CONSTRAINT_IMPLIES_NOEXCEPT
    )

    if(
        NOT SFFDN_HAS_WFUNCTION_EFFECTS
        OR NOT SFFDN_HAS_WFUNCTION_EFFECT_REDECLARATIONS
        OR NOT SFFDN_HAS_WPERF_CONSTRAINT_IMPLIES_NOEXCEPT
    )
        message(FATAL_ERROR "The selected Clang does not support the required function-effect warnings")
    endif()

    message(STATUS "Enabling Clang Function Effect Analysis")
    target_compile_options(
        sfFDN_warnings
        INTERFACE
            -Wfunction-effects
            -Wfunction-effect-redeclarations
            -Wperf-constraint-implies-noexcept
    )

    if(SFFDN_TREAT_FUNCTION_EFFECTS_AS_ERRORS)
        target_compile_options(
            sfFDN_warnings
            INTERFACE
                -Werror=function-effects
                -Werror=function-effect-redeclarations
        )
    endif()
endif()
