# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(3rdparty EXCLUDE_FROM_ALL)
add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
