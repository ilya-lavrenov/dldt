# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wno-error=sign-compare -Wno-error=reorder
                          -Wno-error=unused-function)
endif()

set(IE_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(ngraph_functions)
add_subdirectory(lpt_ngraph_functions)

if(ENABLE_TESTS)
    add_subdirectory(common_test_utils)
    add_subdirectory(unit_test_utils)
endif()

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(shared_test_classes)
    add_subdirectory(functional_test_utils)
    add_subdirectory(plugin_func_tests)
endif()
