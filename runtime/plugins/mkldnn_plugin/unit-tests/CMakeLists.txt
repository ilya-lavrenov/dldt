# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# because unit tests use plugins object files compiled with LTO
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

set(TARGET_NAME cpuUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${OpenVINO_SOURCE_DIR}/runtime/plugins/mkldnn_plugin/src
            ${OpenVINO_SOURCE_DIR}/runtime/core/transformations/include
        OBJECT_FILES
            $<TARGET_OBJECTS:MKLDNNPlugin_obj>
        LINK_LIBRARIES
            unitTestUtils
            mkldnn
            inference_engine_transformations
            inference_engine_lp_transformations
        ADD_CPPLINT
        LABELS
            CPU
)

ie_faster_build(${TARGET_NAME}
    UNITY
)
