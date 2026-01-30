if(MSVC)
    set(SFFDN_WARNINGS_CXX /W3 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(SFFDN_WARNINGS_CXX
        -Wall
        -Wextra
        -Wpedantic
        -Wno-sign-compare
        -Wno-language-extension-token
        -Wunsafe-buffer-usage
    )
endif()

add_library(sfFDN_warnings INTERFACE)
add_library(sfFDN::sfFDN_warnings ALIAS sfFDN_warnings)
target_compile_options(sfFDN_warnings INTERFACE ${SFFDN_WARNINGS_CXX})
