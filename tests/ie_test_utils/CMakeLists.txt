# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(ngraph_functions)
add_subdirectory(lpt_ngraph_functions)

if(ENABLE_TESTS)
    add_subdirectory(common_test_utils)
    add_subdirectory(unit_test_utils)
endif()

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional_test_utils)
endif()
