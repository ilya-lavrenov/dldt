# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(FRONTEND_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/frontend/frontend_manager/include
)

project (ngraph)

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

if (LINUX)
    include(GNUInstallDirs)
else()
    set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Object code libraries (lib)")
    set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE STRING "C header files (include)")
endif()

# Destinations
set(NGRAPH_INSTALL_LIB "deployment_tools/ngraph/${CMAKE_INSTALL_LIBDIR}")
set(NGRAPH_INSTALL_INCLUDE "deployment_tools/ngraph/${CMAKE_INSTALL_INCLUDEDIR}")

#-----------------------------------------------------------------------------------------------
# External projects install directory
#-----------------------------------------------------------------------------------------------

add_subdirectory(test)

