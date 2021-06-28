# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# because unit tests use plugins object files compiled with LTO
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

set(TARGET_NAME gnaUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            unitTestUtils
            GNAPlugin_test_static
        ADD_CPPLINT
        LABELS
            GNA
)