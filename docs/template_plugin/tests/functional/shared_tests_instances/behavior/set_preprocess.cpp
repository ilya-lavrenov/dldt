// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi-device/multi_device_config.hpp"

#include "behavior/set_preprocess.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

const std::vector<std::map<std::string, std::string>> multiConfigs = {
        {{ InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
           CommonTestUtils::DEVICE_TEMPLATE }}
};

const std::vector<std::map<std::string, std::string>> heteroConfigs = {
        {{ "TARGET_FALLBACK", CommonTestUtils::DEVICE_TEMPLATE }}
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, PreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
                        PreprocessTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, PreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(multiConfigs)),
                        PreprocessTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Hetero_BehaviorTests, PreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                ::testing::ValuesIn(heteroConfigs)),
                        PreprocessTest::getTestCaseName);

}  // namespace