if(NOT DEFINED SFFDN_SOURCE_DIR)
    message(FATAL_ERROR "SFFDN_SOURCE_DIR is required")
endif()

file(GLOB_RECURSE realtime_sources
     "${SFFDN_SOURCE_DIR}/src/*.cpp"
     "${SFFDN_SOURCE_DIR}/src/*.h"
)

set(fea_waiver_count 0)
set(rtsan_disabler_count 0)
set(exception_files)

foreach(source IN LISTS realtime_sources)
    file(READ "${source}" contents)

    string(REGEX MATCHALL "SFFDN_FEA_UNSAFE\\(" fea_waivers "${contents}")
    list(LENGTH fea_waivers source_fea_waiver_count)
    math(EXPR fea_waiver_count "${fea_waiver_count} + ${source_fea_waiver_count}")

    string(REGEX MATCHALL "SFFDN_RTSAN_SCOPED_DISABLER\\(" rtsan_disablers "${contents}")
    list(LENGTH rtsan_disablers source_rtsan_disabler_count)
    math(EXPR rtsan_disabler_count "${rtsan_disabler_count} + ${source_rtsan_disabler_count}")

    if(source_fea_waiver_count GREATER 0 OR source_rtsan_disabler_count GREATER 0)
        list(APPEND exception_files "${source}")
    endif()
endforeach()

if(NOT fea_waiver_count EQUAL 0)
    message(FATAL_ERROR "Expected 0 FEA waivers, found ${fea_waiver_count}")
endif()

if(NOT rtsan_disabler_count EQUAL 0)
    message(FATAL_ERROR "Expected 0 RTSan disablers, found ${rtsan_disabler_count}")
endif()

if(exception_files)
    message(FATAL_ERROR "Realtime exceptions are not allowed; found: ${exception_files}")
endif()

message(STATUS "Realtime exception inventory: 0 FEA waivers, 0 RTSan disablers")
