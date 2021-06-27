# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_MYRIAD)
    include(cmake/vpu_dependencies.cmake)
endif()

add_subdirectory(3rdparty)
add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(unit-tests)
    add_subdirectory(tests)
endif()
