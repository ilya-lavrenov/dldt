# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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

if(NOT WIN32)
    message(${CMAKE_CURRENT_SOURCE_DIR}/cmake_static_protobuf)
    message(BINARY ${CMAKE_CURRENT_BINARY_DIR})

    # There seems no suitable other way to identify exact output binary name for libprotobuf
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # Use 'protobufd' directly as it is done in the same way in protobuf cmake files
        set(PROTOBUF_STATIC_LIB_OUTPUT
                ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}protobufd${CMAKE_STATIC_LIBRARY_SUFFIX})
    else(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(PROTOBUF_STATIC_LIB_OUTPUT ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}protobuf${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()

    message("Static protobuf lib: ${PROTOBUF_STATIC_LIB_OUTPUT}")

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libprotobuf_static)
    add_custom_command(
            OUTPUT
            ${PROTOBUF_STATIC_LIB_OUTPUT}
            COMMAND ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_static_protobuf
                    -DOpenVINO_SOURCE_DIR=${OpenVINO_SOURCE_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON # relocation R_X86_64_TPOFF32
                    -DCMAKE_POLICY_DEFAULT_CMP0063=NEW # Honor visibility properties for all target types
                    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
                    -DCMAKE_COMPILE_PDB_OUTPUT_DIRECTORY=${CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY}
                    -DCMAKE_PDB_OUTPUT_DIRECTORY=${CMAKE_PDB_OUTPUT_DIRECTORY}
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_CXX_VISIBILITY_PRESET=${CMAKE_CXX_VISIBILITY_PRESET}
                    -DNGRAPH_INSTALL_LIB=${NGRAPH_INSTALL_LIB}
                    -DSYSTEM_PROTOC=${SYSTEM_PROTOC}
                     ${NGRAPH_FORWARD_CMAKE_ARGS}
            COMMAND ${CMAKE_COMMAND} --build . --target libprotobuf -j8
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libprotobuf_static
            COMMENT "Build Protobuf Static Library"
    )

    add_custom_target(libprotobuf_static
            DEPENDS ${PROTOBUF_STATIC_LIB_OUTPUT})
endif()

set(FRONTEND_INSTALL_INCLUDE "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend")

add_subdirectory(frontend_manager)

if (NGRAPH_ONNX_IMPORT_ENABLE)
    add_subdirectory(onnx_common)
    add_subdirectory(onnx_import)
endif()

if (NGRAPH_ONNX_EDITOR_ENABLE)
    add_subdirectory(onnx_editor)
endif()

if (NGRAPH_PDPD_FRONTEND_ENABLE)
    add_subdirectory(paddlepaddle)
endif()
