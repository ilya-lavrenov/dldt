# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(preprocessing)
add_subdirectory(readers)
add_subdirectory(legacy_api)
add_subdirectory(transformations)
add_subdirectory(inference_engine)
add_subdirectory(low_precision_transformations)
add_subdirectory(offline_transformations)
add_subdirectory(snippets)

# add a custom target to build all Inference Engine Core libraries

add_custom_target(ie_libraries ALL
    DEPENDS inference_engine_transformations inference_engine_legacy
            inference_engine inference_engine_preproc
            inference_engine_ir_v7_reader inference_engine_ir_reader
            inference_engine_lp_transformations inference_engine_snippets)

if(NGRAPH_ONNX_IMPORT_ENABLE)
    add_dependencies(ie_libraries inference_engine_onnx_reader)
endif()
