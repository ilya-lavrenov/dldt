# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_COMPILER_IS_GNUCC)
    ie_add_compiler_flags(-Wall)
endif()

# TODO: remove
set(NGRAPH_TARGETS_FILE "${CMAKE_CURRENT_BINARY_DIR}/ngraphTargets.cmake")
set(NGRAPH_INSTALL_LIB "deployment_tools/ngraph/${CMAKE_INSTALL_LIBDIR}")
set(NGRAPH_INSTALL_INCLUDE "deployment_tools/ngraph/${CMAKE_INSTALL_INCLUDEDIR}")

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(plugins)
add_subdirectory(frontends)
