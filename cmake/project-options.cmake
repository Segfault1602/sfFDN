add_library(sfFDN_options INTERFACE)
add_library(sfFDN::sfFDN_options ALIAS sfFDN_options)
target_compile_features(sfFDN_options INTERFACE cxx_std_23)

# if(MSVC)

# elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
#     target_compile_options(sfFDN_options INTERFACE -march=native)
# endif()

if(SFFDN_USE_SANITIZER)
    message(STATUS "Enabling AddressSanitizer")
    target_compile_options(sfFDN_options INTERFACE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_link_options(sfFDN_options INTERFACE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
endif()

if(SFFDN_USE_RTSAN)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        message(FATAL_ERROR "SFFDN_USE_RTSAN requires a Clang compiler with RealtimeSanitizer support")
    endif()

    include(CheckCXXCompilerFlag)
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    unset(SFFDN_HAS_RTSAN CACHE)
    unset(SFFDN_HAS_FRAME_POINTERS CACHE)
    check_cxx_compiler_flag("-fno-omit-frame-pointer" SFFDN_HAS_FRAME_POINTERS)

    if(NOT SFFDN_HAS_FRAME_POINTERS)
        message(
            FATAL_ERROR
            "SFFDN_USE_RTSAN requires compiler support for '-fno-omit-frame-pointer'"
        )
    endif()

    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=realtime -fno-omit-frame-pointer")
    set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=realtime -fno-omit-frame-pointer)
    check_cxx_source_compiles(
        "void process() [[clang::nonblocking]] {}\nint main() { process(); return 0; }"
        SFFDN_HAS_RTSAN
    )
    cmake_pop_check_state()

    if(NOT SFFDN_HAS_RTSAN)
        message(
            FATAL_ERROR
                "SFFDN_USE_RTSAN requires Clang support for compiling and linking with '-fsanitize=realtime -fno-omit-frame-pointer'; found ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        )
    endif()

    message(STATUS "Enabling Clang RealtimeSanitizer")
    target_compile_options(sfFDN_options INTERFACE -fsanitize=realtime -fno-omit-frame-pointer)
    target_link_options(sfFDN_options INTERFACE -fsanitize=realtime -fno-omit-frame-pointer)
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

include(CheckIncludeFile)
check_include_file(xmmintrin.h HAVE_XMMINTRIN_H)
if(HAVE_XMMINTRIN_H)
    target_compile_definitions(sfFDN_options INTERFACE -DHAVE_XMMINTRIN_H)
endif()

if(SFFDN_USE_VDSP)
    message(STATUS "Enabling vDSP support")
    target_compile_definitions(sfFDN_options INTERFACE -DSFFDN_USE_VDSP)
endif()
