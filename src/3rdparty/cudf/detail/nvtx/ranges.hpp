// Stubbed ranges.hpp - NVTX disabled to avoid conflicts with CUDA 12.8
#pragma once

namespace cudf {
struct libcudf_domain {
    static constexpr char const* name{"libcudf"};
};
}

// No-op macro
#define CUDF_FUNC_RANGE() do {} while(0)
