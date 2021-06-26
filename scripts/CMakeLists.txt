# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_shellcheck_process(DIRECTORY "${OpenVINO_MAIN_SOURCE_DIR}"
                      SKIP "${OpenVINO_MAIN_SOURCE_DIR}/bin"
                           "${OpenVINO_MAIN_SOURCE_DIR}/build"
                           "${OpenVINO_MAIN_SOURCE_DIR}/thirdparty"
                           "${IE_MAIN_SOURCE_DIR}/tests/ie_test_utils/common_test_utils/gtest"
                           "${IE_MAIN_SOURCE_DIR}/samples/thirdparty"
                           "${IE_MAIN_SOURCE_DIR}/thirdparty"
                           "${IE_MAIN_SOURCE_DIR}/temp"
                           # TODO fix and enable back:
                           "${OpenVINO_MAIN_SOURCE_DIR}/scripts/dependencies.sh"
                           "${OpenVINO_MAIN_SOURCE_DIR}/scripts/install_dependencies/install_NEO_OCL_driver.sh"
                           "${OpenVINO_MAIN_SOURCE_DIR}/scripts/install_dependencies/install_openvino_dependencies.sh"
                           "${OpenVINO_MAIN_SOURCE_DIR}/ngraph/python/tests/test_onnx/model_zoo_preprocess.sh"
                           )

# install setupvars

ie_cpack_add_component(setupvars REQUIRED)

if(UNIX)
    install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/setupvars/setupvars.sh
            DESTINATION bin
            COMPONENT setupvars)
elseif(WIN32)
    install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/setupvars/setupvars.bat
            DESTINATION bin
            COMPONENT setupvars)
endif()

# install install_dependencies

if(UNIX)
    ie_cpack_add_component(install_dependencies REQUIRED)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/install_dependencies/
            DESTINATION install_dependencies
            COMPONENT install_dependencies)
endif()

# install files for demo

ie_cpack_add_component(demo_scripts DEPENDS core)

if(UNIX)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/demo/
            DESTINATION deployment_tools/demo
            COMPONENT demo_scripts
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE)
elseif(WIN32)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/demo/
            DESTINATION deployment_tools/demo
            COMPONENT demo_scripts
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE)
endif()
