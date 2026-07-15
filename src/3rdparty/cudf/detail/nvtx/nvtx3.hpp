// Stubbed out nvtx3.hpp to avoid conflicts with CUDA 12.8 built-in NVTX3
// This removes profiling markers but fixes compilation with newer CUDA

#pragma once

#define NVTX3_MINOR_VERSION 0

namespace nvtx3 {

// Stub domain
struct domain {
    struct global {};
    template<typename D>
    static void* get() { return nullptr; }
};

// Stub color types
struct color {
    constexpr color(uint32_t) {}
};
struct argb : color { using color::color; };
struct rgb : color { using color::color; };

// Stub category
template<typename D = domain::global>
struct category {
    constexpr category(uint32_t) {}
};
template<typename D = domain::global>
struct named_category : category<D> {
    constexpr named_category(uint32_t id, const char*) : category<D>(id) {}
};

// Stub message types
struct message { message(const char*) {} };
template<typename D = domain::global>
struct registered_message {
    registered_message(const char*) {}
};

// Stub payload
struct payload {
    payload(int64_t) {}
    payload(double) {}
};

// Stub event_attributes
struct event_attributes {
    event_attributes() = default;
    template<typename... Args>
    event_attributes(Args&&...) {}
    void* get() const { return nullptr; }
};

// Stub ranges
template<typename D = domain::global>
struct domain_thread_range {
    domain_thread_range() = default;
    template<typename... Args>
    domain_thread_range(Args&&...) {}
};

template<typename D = domain::global>
struct domain_process_range {
    domain_process_range() = default;
    template<typename... Args>
    domain_process_range(Args&&...) {}
};

using thread_range = domain_thread_range<>;
using process_range = domain_process_range<>;

// Stub mark function
template<typename D = domain::global>
inline void mark(event_attributes const&) noexcept {}

}  // namespace nvtx3

// Stub out the macros
#define NVTX3_FUNC_RANGE_IN(domain) \
    static ::nvtx3::registered_message<domain> const nvtx3_func_name__{__func__}; \
    static ::nvtx3::event_attributes const nvtx3_func_attr__{nvtx3_func_name__}; \
    ::nvtx3::domain_thread_range<domain> const nvtx3_range__{nvtx3_func_attr__};

#define NVTX3_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(::nvtx3::domain::global)
