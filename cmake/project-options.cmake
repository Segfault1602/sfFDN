add_library(sfFDN_options INTERFACE)
add_library(sfFDN::sfFDN_options ALIAS sfFDN_options)
target_compile_features(sfFDN_options INTERFACE cxx_std_23)

if(MSVC)

elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    target_compile_options(sfFDN_options INTERFACE -march=native)
endif()

if(SFFDN_USE_SANITIZER)
    if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        message(STATUS "Enabling AddressSanitizer")
        target_compile_options(sfFDN_options INTERFACE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
        target_link_options(sfFDN_options INTERFACE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    endif()
endif()

include(CheckCXXSymbolExists)

if(cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(header version)
else()
    set(header ciso646)
endif()

check_cxx_symbol_exists(_LIBCPP_VERSION ${header} LIBCPP)
if(LIBCPP)
    if(SFFDN_ENABLE_HARDENING)
        message(STATUS "Enabling libc++ hardening")
        target_compile_definitions(
            sfFDN_options INTERFACE $<$<CONFIG:Debug>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG>
                                    $<$<CONFIG:RelWithDebInfo>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG>
        )
    endif()
endif()

check_cxx_symbol_exists(_STD_VERSION_HEADER_ ${header} MSVC_STL)
if(MSVC_STL)
    if(SFFDN_ENABLE_HARDENING)
        message(STATUS "Enabling MSVC STL hardening")
        target_compile_definitions(
            sfFDN_options INTERFACE $<$<CONFIG:Debug>:_MSVC_STL_HARDENING=1>
                                    $<$<CONFIG:RelWithDebInfo>:_MSVC_STL_HARDENING=1>
        )
    endif()
endif()

if(APPLE)
    set(SFFDN_USE_VDSP ON)
    target_compile_definitions(sfFDN_options INTERFACE -DSFFDN_USE_VDSP)
endif()
