// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#if defined(__clang__)
#if __has_attribute(nonblocking)
#define SFFDN_NONBLOCKING [[clang::nonblocking]]
#else
#define SFFDN_NONBLOCKING
#endif
#else
#define SFFDN_NONBLOCKING
#endif

#if defined(__clang__)
#define SFFDN_FEA_UNSAFE(...)                                                                                          \
    _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wunknown-warning-option\"")                  \
        _Pragma("clang diagnostic ignored \"-Wfunction-effects\"") __VA_ARGS__ _Pragma("clang diagnostic pop")
#else
#define SFFDN_FEA_UNSAFE(...) __VA_ARGS__
#endif

#if defined(__has_feature)
#if __has_feature(realtime_sanitizer)
#include <sanitizer/rtsan_interface.h>
#define SFFDN_RTSAN_SCOPED_DISABLER(name) __rtsan::ScopedDisabler name
#else
#define SFFDN_RTSAN_SCOPED_DISABLER(name)
#endif
#else
#define SFFDN_RTSAN_SCOPED_DISABLER(name)
#endif
