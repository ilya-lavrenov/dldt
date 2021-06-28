# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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

add_subdirectory(oneDNN EXCLUDE_FROM_ALL)
add_library(mkldnn ALIAS dnnl)

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND (MSVC_VERSION VERSION_GREATER_EQUAL "1910"))
    # 1910 version of Visual Studio 2017
    # This flagis needed for enabling SIMD vectorization with command '#pragma omp simd'.
    # Compilation with '/openmp:experimental' key allow us to enable vectorizatikon capability in MSVC.
    target_compile_options(mkldnn PRIVATE "/openmp:experimental")
endif()
