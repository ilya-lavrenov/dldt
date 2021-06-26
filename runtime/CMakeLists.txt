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
set(NGRAPH_INSTALL_DOC "deployment_tools/ngraph/${CMAKE_INSTALL_DOCDIR}")
set(NGRAPH_INSTALL_BIN "deployment_tools/ngraph/${CMAKE_INSTALL_BINDIR}")

set(USE_STATIC_PROTOBUF OFF)
if (NGRAPH_PDPD_FRONTEND_ENABLE) # add more frontends here which depend on static protobuf
    set(USE_STATIC_PROTOBUF ON)
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE OR USE_STATIC_PROTOBUF)
    if (MSVC)
        # When we build dll libraries. These flags make sure onnx and protobuf build with /MD, not /MT.
        # These two options can't be mixed, because they requires link two imcompatiable runtime.
        set(protobuf_WITH_ZLIB  OFF CACHE BOOL "" FORCE)

        if(NOT DEFINED ONNX_USE_MSVC_STATIC_RUNTIME)
            set(ONNX_USE_MSVC_STATIC_RUNTIME OFF)
        endif()
        if(NOT DEFINED protobuf_MSVC_STATIC_RUNTIME)
            set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Link protobuf to static runtime libraries" FORCE)
        endif()
    endif()

    set(BEFORE_ONNX_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS ON)
    set(BUILD_STANDALONE_STATIC OFF)

    if (NOT NGRAPH_USE_SYSTEM_PROTOBUF)
        include(${OpenVINO_SOURCE_DIR}/ngraph/cmake/external_protobuf.cmake)
    else()
        find_package(Protobuf 2.6.1 REQUIRED)
    endif()

    if (NGRAPH_ONNX_IMPORT_ENABLE)
        # target onnx_proto will be shared lib, onnx static
        include(${OpenVINO_SOURCE_DIR}/ngraph/cmake/external_onnx.cmake)
        if (TARGET ext_protobuf)
            add_dependencies(onnx ext_protobuf)
        endif()
    endif()

    set(BUILD_SHARED_LIBS ${BEFORE_ONNX_BUILD_SHARED_LIBS})
    unset(BEFORE_ONNX_BUILD_SHARED_LIBS)
    unset(BUILD_STANDALONE_STATIC)
endif()

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(plugins)
add_subdirectory(frontends)
