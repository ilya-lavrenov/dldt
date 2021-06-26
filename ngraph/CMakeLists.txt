# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(FRONTEND_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/frontend/frontend_manager/include
)

# Will be used by frontends to construct frontend-specific source location paths
set(FRONTEND_BASE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/frontend
)

project (ngraph)

option(NGRAPH_UNIT_TEST_ENABLE "Control the building of unit tests" ON)
option(NGRAPH_UNIT_TEST_BACKENDS_ENABLE "Control the building of unit tests using backends" ON)
option(NGRAPH_INTERPRETER_ENABLE "Control the building of the INTERPRETER backend" ON)
option(NGRAPH_DEBUG_ENABLE "Enable output for NGRAPH_DEBUG statements" OFF)
option(NGRAPH_ONNX_IMPORT_ENABLE "Enable ONNX importer" OFF)
option(NGRAPH_ONNX_EDITOR_ENABLE "Enable ONNX Editor" OFF)
option(NGRAPH_PDPD_FRONTEND_ENABLE "Enable PaddlePaddle FrontEnd" OFF)
option(NGRAPH_PYTHON_BUILD_ENABLE "Enable build nGraph python package wheel" OFF)
option(NGRAPH_DYNAMIC_COMPONENTS_ENABLE "Enable dynamic loading of components" ON)
option(NGRAPH_USE_PROTOBUF_LITE "Compiles and links with protobuf-lite" OFF)

if (NGRAPH_ONNX_IMPORT_ENABLE OR NGRAPH_PDPD_FRONTEND_ENABLE)
    option(NGRAPH_USE_SYSTEM_PROTOBUF "Use system provided Protobuf shared object" OFF)
endif()
if(NGRAPH_ONNX_EDITOR_ENABLE AND NOT NGRAPH_ONNX_IMPORT_ENABLE)
    message(FATAL_ERROR "ONNX Editor component requires ONNX Importer. Set NGRAPH_ONNX_IMPORT_ENABLE=ON.")
endif()

message(STATUS "NGRAPH_DEBUG_ENABLE:                  ${NGRAPH_DEBUG_ENABLE}")
message(STATUS "NGRAPH_DYNAMIC_COMPONENTS_ENABLE:     ${NGRAPH_DYNAMIC_COMPONENTS_ENABLE}")
message(STATUS "NGRAPH_INTERPRETER_ENABLE:            ${NGRAPH_INTERPRETER_ENABLE}")
message(STATUS "NGRAPH_ONNX_IMPORT_ENABLE:            ${NGRAPH_ONNX_IMPORT_ENABLE}")
message(STATUS "NGRAPH_ONNX_EDITOR_ENABLE:            ${NGRAPH_ONNX_EDITOR_ENABLE}")
message(STATUS "NGRAPH_PDPD_FRONTEND_ENABLE:          ${NGRAPH_PDPD_FRONTEND_ENABLE}")
message(STATUS "NGRAPH_PYTHON_BUILD_ENABLE:           ${NGRAPH_PYTHON_BUILD_ENABLE}")
message(STATUS "NGRAPH_USE_PROTOBUF_LITE:             ${NGRAPH_USE_PROTOBUF_LITE}")
message(STATUS "NGRAPH_UNIT_TEST_ENABLE:              ${NGRAPH_UNIT_TEST_ENABLE}")
message(STATUS "NGRAPH_UNIT_TEST_BACKENDS_ENABLE:     ${NGRAPH_UNIT_TEST_BACKENDS_ENABLE}")

# Setup CMAKE_ARGS to be forwarded to External Projects
set(NGRAPH_FORWARD_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
    -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=${CMAKE_EXPORT_COMPILE_COMMANDS}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${CMAKE_POSITION_INDEPENDENT_CODE}
    -DCMAKE_DEBUG_POSTFIX=${CMAKE_DEBUG_POSTFIX}
    -DCMAKE_RELEASE_POSTFIX=${CMAKE_RELEASE_POSTFIX}
    )

if(CMAKE_TOOLCHAIN_FILE)
    set(NGRAPH_FORWARD_CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            )
endif()

if (CMAKE_OSX_SYSROOT)
    set(NGRAPH_FORWARD_CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
            )
endif()

if (NOT MSVC)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    endif()

    set(NGRAPH_FORWARD_CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            )
endif()
message(STATUS "NGRAPH_FORWARD_CMAKE_ARGS ${NGRAPH_FORWARD_CMAKE_ARGS}")

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

if (LINUX)
    include(GNUInstallDirs)
else()
    set(CMAKE_INSTALL_BINDIR "bin" CACHE STRING "User executables (bin)")
    set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Object code libraries (lib)")
    set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE STRING "C header files (include)")
    set(CMAKE_INSTALL_DOCDIR "doc" CACHE STRING "Document files (doc)")
    mark_as_advanced(CMAKE_INSTALL_BINDIR CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR, CMAKE_INSTALL_DOCDIR)
endif()

if (DEFINED NGRAPH_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${NGRAPH_INSTALL_PREFIX})
endif()
message(STATUS "Installation directory: ${CMAKE_INSTALL_PREFIX}")

# Destinations
set(NGRAPH_INSTALL_LIB "deployment_tools/ngraph/${CMAKE_INSTALL_LIBDIR}")
set(NGRAPH_INSTALL_INCLUDE "deployment_tools/ngraph/${CMAKE_INSTALL_INCLUDEDIR}")
set(NGRAPH_INSTALL_DOC "deployment_tools/ngraph/${CMAKE_INSTALL_DOCDIR}")
set(NGRAPH_INSTALL_BIN "deployment_tools/ngraph/${CMAKE_INSTALL_BINDIR}")

if (NGRAPH_DYNAMIC_COMPONENTS_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNGRAPH_DYNAMIC_COMPONENTS_ENABLE")
endif()

#-----------------------------------------------------------------------------------------------
# External projects install directory
#-----------------------------------------------------------------------------------------------

# Build destination directory for nGraph binaries and tools.

if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
    set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_subdirectory(test)

if(NGRAPH_PYTHON_BUILD_ENABLE)
    add_subdirectory(python)
endif()
