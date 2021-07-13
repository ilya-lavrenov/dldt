// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "permute_params.h"
#include <vector>

namespace kernel_selector {
class PermuteKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~PermuteKernelBase() {}

    bool Validate(const Params& p, const optional_params& o) const override;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
protected:
    JitConstants GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const;
    virtual CommonDispatchData SetDefault(const permute_params& params) const = 0;
};
}  // namespace kernel_selector
