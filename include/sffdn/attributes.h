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
