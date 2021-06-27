# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(c)

if(ENABLE_PYTHON)
    add_subdirectory(python_ngraph)
    add_subdirectory(python_ie)
endif()
