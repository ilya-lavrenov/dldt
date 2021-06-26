# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

add_subdirectory(ittapi EXCLUDE_FROM_ALL)
add_subdirectory(itt_collector EXCLUDE_FROM_ALL)
add_subdirectory(zlib EXCLUDE_FROM_ALL)
add_subdirectory(cnpy EXCLUDE_FROM_ALL)
add_subdirectory(ocl EXCLUDE_FROM_ALL)

add_subdirectory(xbyak EXCLUDE_FROM_ALL)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS xbyak)

#
# Pugixml
#

if(ENABLE_SYSTEM_PUGIXML)
    find_package(PugiXML REQUIRED)
    set_property(TARGET pugixml PROPERTY IMPORTED_GLOBAL TRUE)
    add_library(pugixml::static ALIAS pugixml)
else()
    function(ie_build_pugixml)
        set(BUILD_SHARED_LIBS OFF)
        add_subdirectory(pugixml EXCLUDE_FROM_ALL)
        set_property(TARGET pugixml-static PROPERTY EXPORT_NAME pugixml)
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS pugixml::static)
    endfunction()

    ie_build_pugixml()
endif()

#
# Fluid, G-API, OpenCV HAL
#

add_library(ocv_hal INTERFACE)
target_include_directories(ocv_hal INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/ocv")

add_subdirectory(ade EXCLUDE_FROM_ALL)
add_subdirectory(fluid/modules/gapi EXCLUDE_FROM_ALL)
set_target_properties(ade fluid PROPERTIES FOLDER thirdparty)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS ade fluid)

#
# Gflags
#

function(add_gflags)
    # common gflags settings
    set(GFLAGS_IS_SUBPROJECT TRUE)
    set(HAVE_SYS_STAT_H 1)
    set(HAVE_INTTYPES_H 1)
    set(INTTYPES_FORMAT C99)
    set(BUILD_TESTING OFF)
    set(BUILD_SHARED_LIBS OFF)

    add_subdirectory(gflags EXCLUDE_FROM_ALL)
    set_target_properties(gflags_nothreads_static PROPERTIES FOLDER thirdparty)
    openvino_developer_export_targets(COMPONENT openvino_common TARGETS gflags)
endfunction()

add_gflags()

#
# Google Tests framework
#

function(add_gtest_libraries)
    set(gtest_force_shared_crt ON CACHE BOOL "disable static CRT for google test")
    if (UNIX)
        ie_add_compiler_flags(-Wno-undef)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            ie_add_compiler_flags(-Wno-deprecated-copy)
        endif()
    endif()

    set(BUILD_SHARED_LIBS OFF)
    set(INSTALL_GTEST OFF)
    add_subdirectory(gtest EXCLUDE_FROM_ALL)

    get_target_property(gtest_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gtest_include_dirs}")

    get_target_property(gmock_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gmock PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gmock_include_dirs};${gmock_SOURCE_DIR}/include")

    # If we have specified /Z7 option, remove -Zi option which comes from gtest
    if (WIN32)
        set(gtest_targets "gtest;gtest_main;gmock;gmock_main")
        foreach(target_name ${gtest_targets})
            if(TARGET "${target_name}")
                get_target_property(_target_cxx_flags ${target_name} COMPILE_OPTIONS)
                if(_target_cxx_flags)
                    if(CMAKE_CXX_FLAGS_DEBUG MATCHES ".+/Z7.+" OR CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES ".+/Z7.+")
                        string(REPLACE "-Zi" " " _target_cxx_flags ${_target_cxx_flags})
                        message(STATUS "Removing -Zi flag from target " ${target_name})
                        set_target_properties(${target_name} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
                    endif()
                endif()
            endif()
        endforeach()
    endif()
    set_target_properties(gtest gtest_main gmock gmock_main
                          PROPERTIES FOLDER thirdparty)
endfunction()

if(NGRAPH_UNIT_TEST_ENABLE OR ENABLE_TESTS)
    add_gtest_libraries()
endif()

#
# ClDNN
#

if(ENABLE_CLDNN)
    if(ENABLE_TESTS AND ENABLE_CLDNN_TESTS)
        set(CLDNN__INCLUDE_TESTS ON CACHE BOOL "" FORCE)
    else()
        set(CLDNN__INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    endif()
    set(CLDNN_THREADING "${THREADING}" CACHE STRING "" FORCE)
    set(GPU_DEBUG_CONFIG OFF CACHE BOOL "Enable debug config feature" FORCE)

    add_subdirectory(clDNN)
endif()

#
# OneDNN for Intel CPU plugin
#

if(ENABLE_MKL_DNN)
    set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE_CACHE OFF CACHE BOOL "" FORCE) ## TODO: try it later
    set(DNNL_ENABLE_MAX_CPU_ISA OFF CACHE BOOL "" FORCE)     ## TODO: try it later
    set(DNNL_LIBRARY_TYPE STATIC CACHE BOOL "" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DNNL_CPU_RUNTIME "${THREADING}" CACHE BOOL "" FORCE)
    set(DNNL_BLAS_VENDOR "NONE" CACHE BOOL "" FORCE)
    set(SDL_cmake_included ON)  ## to skip internal SDL flags. SDL flags are already set on IE level
    if (ANDROID OR ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND NOT (THREADING STREQUAL "OMP")))
        set(OpenMP_cmake_included ON) ## to skip "omp simd" inside a code. Lead to some crashes inside NDK LLVM..
    endif()

    add_subdirectory(onednn-cpu EXCLUDE_FROM_ALL)
    add_library(mkldnn ALIAS dnnl)

    if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND (MSVC_VERSION VERSION_GREATER_EQUAL "1910"))
        # 1910 version of Visual Studio 2017
        # This flagis needed for enabling SIMD vectorization with command '#pragma omp simd'.
        # Compilation with '/openmp:experimental' key allow us to enable vectorizatikon capability in MSVC.
        target_compile_options(mkldnn PRIVATE "/openmp:experimental")
    endif()
endif()

#
# Install
#

ie_cpack_add_component(cpp_samples_deps)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gflags
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zlib
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cnpy
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)
