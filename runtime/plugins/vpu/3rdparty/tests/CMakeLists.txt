# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_COMPILER_IS_GNUCC)
    ie_add_compiler_flags(-Wno-error=sign-compare)
endif()

add_subdirectory(XLink)
