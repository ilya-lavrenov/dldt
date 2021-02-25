﻿// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <algorithm>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include <legacy/details/ie_cnn_network_tools.h>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "functional_test_utils/network_utils.hpp"

#include <ie_plugin_ptr.hpp>
#include <common_test_utils/test_constants.hpp>
#include "details/ie_so_loader.h"
#include "ie_metric_helpers.hpp"


#include "unit_test_utils/mocks/mock_engine/mock_plugin.hpp"
#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>


using namespace std;
using namespace InferenceEngine;
using namespace ::testing;
using namespace InferenceEngine::details;

class CachingInferencePlugin;

class DummyExecutableNetwork : public ExecutableNetworkInternal {
    CachingInferencePlugin& m_plugin;
public:
    DummyExecutableNetwork(CachingInferencePlugin& plugin) : m_plugin(plugin) {}
    void ExportImpl(std::ostream& networkModel) override;
    IInferRequest::Ptr CreateInferRequest() override {
        return nullptr;
    }
};

class CachingInferencePlugin : public InferenceEngine::InferencePluginInternal {
public:
    int m_loadNetworkCalled = 0;
    int m_importNetworkCalled = 0;
    mutable int m_getMetricCalled = 0;
    int m_exportCalled = 0;
    bool m_supportImportExport = true;

    void resetCounters() {
        m_loadNetworkCalled = 0;
        m_importNetworkCalled = 0;
        m_getMetricCalled = 0;
        m_exportCalled = 0;
    }

    ~CachingInferencePlugin() override = default;

    ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const CNNNetwork& network,
        const std::map<std::string, std::string>& config) override {
        m_loadNetworkCalled++;
        return std::make_shared<DummyExecutableNetwork>(*this);
    }

    ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const CNNNetwork& network, RemoteContext::Ptr context,
        const std::map<std::string, std::string>& config) override {
        m_loadNetworkCalled++;
        return std::make_shared<DummyExecutableNetwork>(*this);
    }

    ExecutableNetwork ImportNetworkImpl(std::istream& networkModel,
        const std::map<std::string, std::string>& config) override {
        m_importNetworkCalled++;
        return ExecutableNetwork(std::make_shared<MockIExecutableNetwork>());
    }

    ExecutableNetwork ImportNetworkImpl(std::istream& networkModel,
        const RemoteContext::Ptr& context,
        const std::map<std::string, std::string>& config) override {
        m_importNetworkCalled++;
        return ExecutableNetwork(std::make_shared<MockIExecutableNetwork>());
    }

    Parameter GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override {
        if (METRIC_KEY(SUPPORTED_METRICS) == name) {
            std::vector<std::string> supportedMetrics = {
                METRIC_KEY(IMPORT_EXPORT_SUPPORT)
            };
            return supportedMetrics;
        } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
            m_getMetricCalled++;
            return m_supportImportExport;
        } else {
            THROW_IE_EXCEPTION << "Unsupported device metric: " << name;
        }
    }
};

void DummyExecutableNetwork::ExportImpl(std::ostream& networkModel) {
    networkModel << "Export is called";
    m_plugin.m_exportCalled++;
}

//------------------------------------------------------

class CachingTest : public ::testing::Test {
public:
    unique_ptr<SharedObjectLoader> sharedObjectLoader;
    std::function<void(IInferencePlugin*)> injectProxyEngine;
    std::shared_ptr<CachingInferencePlugin> m_plugin; // must be shared_ptr due to internal InferencePluginInternal logic
    std::string modelName = "Caching_test.xml";
    std::string weightsName = "Caching_test.bin";
    std::string deviceName = "mock";
    InferenceEngine::Core ie;

    std::string get_mock_engine_name() {
        std::string mockEngineName("mock_engine");
        return CommonTestUtils::pre + mockEngineName + IE_BUILD_POSTFIX + CommonTestUtils::ext;
    }

    void SetUp() override {
        CommonTestUtils::makeDir("testCache");
        m_plugin = std::make_shared<CachingInferencePlugin>();
        std::string libraryName = get_mock_engine_name();
        sharedObjectLoader.reset(new SharedObjectLoader(libraryName.c_str()));
        injectProxyEngine = make_std_function<void(IInferencePlugin*)>("InjectProxyEngine");
        injectProxyEngine(m_plugin.get());

        FuncTestUtils::TestModel::generateTestModel(modelName, weightsName);

        ie.RegisterPlugin(std::string("mock_engine") + IE_BUILD_POSTFIX, deviceName);
    }

    void TearDown() override {
        m_plugin = nullptr;
        CommonTestUtils::removeIRFiles(modelName, weightsName);
        // remove all cache entries
        CommonTestUtils::removeFilesWithExt("testCache", "blob");
        CommonTestUtils::removeDir("testCache");

        ie.UnregisterPlugin(deviceName);
    }

    void enableCacheConfig() {
        ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), "testCache"} });
    }

private:
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function <T> ptr(reinterpret_cast<T*>(sharedObjectLoader->get_symbol(functionName.c_str())));
        return ptr;
    }
};

TEST_F(CachingTest, Test1) {
    enableCacheConfig();
    auto performReadAndLoad = [&] {
        auto cnnNetwork = ie.ReadNetwork(modelName);
        auto exeNet = ie.LoadNetwork(cnnNetwork, deviceName);
        (void)exeNet;
    };

    { // Step 1: read and load network without cache
        performReadAndLoad();

        EXPECT_GT(m_plugin->m_getMetricCalled, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCalled, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCalled, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCalled, 0); // verify: 'import was not called'
    }

    m_plugin->resetCounters();

    { // Step 2: same load, but now cache must be available from Step 1
        performReadAndLoad();

        EXPECT_GT(m_plugin->m_getMetricCalled, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCalled, 0); // verify: 'load was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_exportCalled, 0); // verify: 'export was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_importNetworkCalled, 1); // verify: 'import was called instead of load + export'
    }
}

TEST_F(CachingTest, TestLoadByName) {
    enableCacheConfig();
    auto performLoadByName = [&] {
        auto exeNet = ie.LoadNetwork(modelName, deviceName);
        (void)exeNet;
    };

    { // Step 1: read and load network without cache
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCalled, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCalled, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCalled, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCalled, 0); // verify: 'import was not called'
    }

    m_plugin->resetCounters();

    { // Step 2: same load, but now cache must be available from Step 1
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCalled, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCalled, 0); // verify: 'load was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_exportCalled, 0); // verify: 'export was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_importNetworkCalled, 1); // verify: 'import was called instead of load + export'
    }
}

TEST_F(CachingTest, TestLoadByName_NoCacheSupported1) {
    enableCacheConfig();                     // Enable caching in global settings
    m_plugin->m_supportImportExport = false; // but it is not supported by plugin

    auto performLoadByName = [&] {
        auto exeNet = ie.LoadNetwork(modelName, deviceName);
        (void)exeNet;
    };

    { // read and load network without cache
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCalled, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCalled, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCalled, 0); // verify: 'export was not called'
        EXPECT_EQ(m_plugin->m_importNetworkCalled, 0); // verify: 'import was not called'
    }
}

TEST_F(CachingTest, TestLoadByName_NoCacheEnabled) {
    auto performLoadByName = [&] {
        auto exeNet = ie.LoadNetwork(modelName, deviceName);
        (void)exeNet;
    };

    { // read and load network without cache
        performLoadByName();

        EXPECT_EQ(m_plugin->m_getMetricCalled, 0); // verify: 'getMetric was not called when global cache is disabled'
        EXPECT_EQ(m_plugin->m_loadNetworkCalled, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCalled, 0); // verify: 'export was not called'
        EXPECT_EQ(m_plugin->m_importNetworkCalled, 0); // verify: 'import was not called'
    }
}