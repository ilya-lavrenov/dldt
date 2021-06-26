# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(OpenVINO)

set(OpenVINO_MAIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(IE_MAIN_SOURCE_DIR ${OpenVINO_MAIN_SOURCE_DIR}/inference-engine)

find_package(IEDevScripts REQUIRED
             PATHS "${OpenVINO_MAIN_SOURCE_DIR}/cmake/developer_package"
             NO_CMAKE_FIND_ROOT_PATH
             NO_DEFAULT_PATH)

include(cmake/features.cmake)

# These options are shared with 3rdparty plugins by means of developer package
include(cmake/dependencies.cmake)

if(ENABLE_COVERAGE)
    include(cmake/coverage.cmake)
endif()

# resolving dependencies for the project
message (STATUS "PROJECT ............................... " ${PROJECT_NAME})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "OpenVINO_MAIN_SOURCE_DIR .............. " ${OpenVINO_MAIN_SOURCE_DIR})
message (STATUS "IE_MAIN_SOURCE_DIR .................... " ${IE_MAIN_SOURCE_DIR})
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})
message (STATUS "CMAKE_BUILD_TYPE ...................... " ${CMAKE_BUILD_TYPE})

# remove file with exported developer targets to force its regeneration
file(REMOVE "${CMAKE_BINARY_DIR}/InferenceEngineTargets.cmake")
foreach(component IN LISTS openvino_export_components)
    file(REMOVE "${CMAKE_BINARY_DIR}/${component}_dev_targets.cmake")
    unset(${component} CACHE)
endforeach()

#
# Build
#

function(build_ngraph)
    function(ngraph_set option value)
        if(NOT DEFINED ${option})
            set(${option} ${value} CACHE BOOL "" FORCE)
        endif()
    endfunction()

    if(ENABLE_TESTS AND NOT ANDROID)
        ngraph_set(NGRAPH_UNIT_TEST_ENABLE ON)
    else()
        ngraph_set(NGRAPH_UNIT_TEST_ENABLE OFF)
    endif()

    if(NOT (ANDROID OR WINDOWS_STORE OR (MSVC AND (ARM OR AARCH64)) ))
        ngraph_set(NGRAPH_ONNX_IMPORT_ENABLE ON)
        ngraph_set(NGRAPH_PDPD_FRONTEND_ENABLE ON)
    else()
        ngraph_set(NGRAPH_ONNX_IMPORT_ENABLE OFF)
        ngraph_set(NGRAPH_PDPD_FRONTEND_ENABLE OFF)
    endif()
    ngraph_set(NGRAPH_INTERPRETER_ENABLE ON)

    if(ENABLE_PYTHON)
        ngraph_set(NGRAPH_PYTHON_BUILD_ENABLE ON)
    else()
        ngraph_set(NGRAPH_PYTHON_BUILD_ENABLE OFF)
    endif()

    if(ENABLE_LTO)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    endif()

    add_subdirectory(ngraph)
    set(NGRAPH_LIBRARIES ngraph PARENT_SCOPE)
endfunction()

function(openvino_developer_export_targets)
    cmake_parse_arguments(EXPORT "" "COMPONENT" "TARGETS" ${ARGN})

    if(EXPORT_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "openvino_developer_export_targets has unparsed arguments: ${EXPORT_UNPARSED_ARGUMENTS}")
    endif()

    set(${EXPORT_COMPONENT} "${${EXPORT_COMPONENT}};${EXPORT_TARGETS}")

    # to allow exporting of aliased targets with the original names
    foreach(target_name IN LISTS ${EXPORT_COMPONENT})
        if(TARGET "${target_name}")
            get_target_property(original_name ${target_name} ALIASED_TARGET)
            if(TARGET "${original_name}")
                message(STATUS "The name ${target_name} is an ALIAS for ${original_name}. "
                        "It will be exported to the InferenceEngineDeveloperPackage with the original name.")
                list(REMOVE_ITEM ${EXPORT_COMPONENT} ${target_name})
                list(APPEND ${EXPORT_COMPONENT} ${original_name})
            endif()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES ${EXPORT_COMPONENT})
    set(${EXPORT_COMPONENT} "${${EXPORT_COMPONENT}}" CACHE INTERNAL
        "A list of OpenVINO ${EXPORT_COMPONENT} exported targets" FORCE)

    list(APPEND openvino_export_components ${EXPORT_COMPONENT})
    list(REMOVE_DUPLICATES openvino_export_components)
    set(openvino_export_components "${openvino_export_components}" CACHE INTERNAL
        "A list of OpenVINO exported components" FORCE)
endfunction()

add_subdirectory(thirdparty)
add_subdirectory(runtime)
add_subdirectory(inference-engine)
build_ngraph()
add_subdirectory(tools)
add_subdirectory(samples)

#
# Developer package
#

openvino_developer_export_targets(COMPONENT openvino_common TARGETS format_reader ie_samples_utils)

# for Template plugin
if(NGRAPH_INTERPRETER_ENABLE)
    openvino_developer_export_targets(COMPONENT ngraph TARGETS ngraph_backend interpreter_backend)
endif()

function(ie_generate_dev_package_config)
    # dummy check that OpenCV is here
    find_package(OpenCV QUIET)

    set(all_dev_targets gflags ie_libraries)
    foreach(component IN LISTS openvino_export_components)
        export(TARGETS ${${component}} NAMESPACE IE::
            APPEND FILE "${CMAKE_BINARY_DIR}/${component}_dev_targets.cmake")
        list(APPEND all_dev_targets ${${component}})
    endforeach()
    add_custom_target(ie_dev_targets ALL DEPENDS ${all_dev_targets})

    configure_package_config_file("${OpenVINO_MAIN_SOURCE_DIR}/cmake/templates/InferenceEngineDeveloperPackageConfig.cmake.in"
                                  "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig.cmake"
                                  INSTALL_DESTINATION share # not used
                                  PATH_VARS "OpenVINO_MAIN_SOURCE_DIR;IE_MAIN_SOURCE_DIR;gflags_BINARY_DIR"
                                  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

    configure_file("${OpenVINO_MAIN_SOURCE_DIR}/cmake/templates/InferenceEngineConfig-version.cmake.in"
                   "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig-version.cmake"
                   @ONLY)
endfunction()

ie_generate_dev_package_config()

#
# Add extra modules
#

function(register_extra_modules)
    # post export
    openvino_developer_export_targets(COMPONENT inference_engine TARGETS inference_engine)
    openvino_developer_export_targets(COMPONENT ngraph TARGETS ngraph)

    set(InferenceEngineDeveloperPackage_DIR "${CMAKE_CURRENT_BINARY_DIR}/build-modules")

    function(generate_fake_dev_package)
        set(iedevconfig_file "${InferenceEngineDeveloperPackage_DIR}/InferenceEngineDeveloperPackageConfig.cmake")
        file(REMOVE "${iedevconfig_file}")

        file(WRITE "${iedevconfig_file}" "\# !! AUTOGENERATED: DON'T EDIT !!\n\n")
        file(APPEND "${iedevconfig_file}" "ie_deprecated_no_errors()\n")

        foreach(target IN LISTS ${openvino_export_components})
            if(target)
                file(APPEND "${iedevconfig_file}" "add_library(IE::${target} ALIAS ${target})\n")
            endif()
        endforeach()
    endfunction()

    generate_fake_dev_package()

    # automatically import plugins from the 'plugins' folder
    file(GLOB local_extra_modules "plugins/*")
    if(NGRAPH_INTERPRETER_ENABLE)
        list(APPEND local_extra_modules "${OpenVINO_MAIN_SOURCE_DIR}/docs/template_plugin")
    endif()

    # detect where IE_EXTRA_MODULES contains folders with CMakeLists.txt
    # other folders are supposed to have sub-folders with CMakeLists.txt
    foreach(module_path IN LISTS IE_EXTRA_MODULES)
        if(EXISTS "${module_path}/CMakeLists.txt")
            list(APPEND extra_modules "${module_path}")
        elseif(module_path)
            file(GLOB extra_modules ${extra_modules} "${module_path}/*")
        endif()
    endforeach()

    # add each extra module
    foreach(module_path IN LISTS extra_modules local_extra_modules)
        if(module_path)
            get_filename_component(module_name "${module_path}" NAME)
            set(build_module ON)
            if(NOT EXISTS "${module_path}/CMakeLists.txt") # if module is built not using cmake
                set(build_module OFF)
            endif()
            if(NOT DEFINED BUILD_${module_name})
                set(BUILD_${module_name} ${build_module} CACHE BOOL "Build ${module_name} extra module" FORCE)
            endif()
            if(BUILD_${module_name})
                message(STATUS "Register ${module_name} to be built in build-modules/${module_name}")
                add_subdirectory("${module_path}" "build-modules/${module_name}")
            endif()
        endif()
    endforeach()
endfunction()

register_extra_modules()

# TODO: move up
add_subdirectory(docs)

#
# Install
#

# install C++ samples

ie_cpack_add_component(cpp_samples DEPENDS cpp_samples_deps core)

if(UNIX)
    install(DIRECTORY samples/
            DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE
            PATTERN speech_libs_and_demos EXCLUDE
            PATTERN .clang-format EXCLUDE)
elseif(WIN32)
    install(DIRECTORY samples/
            DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE
            PATTERN speech_libs_and_demos EXCLUDE
            PATTERN .clang-format EXCLUDE)
endif()

# install C samples

ie_cpack_add_component(c_samples DEPENDS core_c)

if(UNIX)
    install(PROGRAMS samples/build_samples.sh
            DESTINATION ${IE_CPACK_IE_DIR}/samples/c
            COMPONENT c_samples)
elseif(WIN32)
    install(PROGRAMS samples/build_samples_msvc.bat
            DESTINATION ${IE_CPACK_IE_DIR}/samples/c
            COMPONENT c_samples)
endif()

install(DIRECTORY ie_bridges/c/samples/
        DESTINATION ${IE_CPACK_IE_DIR}/samples/c
        COMPONENT c_samples
        PATTERN ie_bridges/c/samples/CMakeLists.txt EXCLUDE
        PATTERN ie_bridges/c/samples/.clang-format EXCLUDE)

install(FILES samples/CMakeLists.txt
        DESTINATION ${IE_CPACK_IE_DIR}/samples/c
        COMPONENT c_samples)

# install speech demo files

if(SPEECH_LIBS_AND_DEMOS)
    ie_cpack_add_component(speech_demo_files)

    install(DIRECTORY ${TEMP}/deployment_tools
                      ${TEMP}/data_processing
            DESTINATION .
            USE_SOURCE_PERMISSIONS
            COMPONENT speech_demo_files)
endif()

ie_cpack(${IE_CPACK_COMPONENTS_ALL})
