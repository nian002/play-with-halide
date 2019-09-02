/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <chrono>

#include "utils.h"


// Prefer high_resolution_clock, but only if it's steady...
template <bool HighResIsSteady = std::chrono::high_resolution_clock::is_steady>
struct SteadyClock {
    using type = std::chrono::high_resolution_clock;
};

// ...otherwise use steady_clock.
template <>
struct SteadyClock<false> {
    using type = std::chrono::steady_clock;
};

inline SteadyClock<>::type::time_point benchmark_now() {
    return SteadyClock<>::type::now();
}

inline double benchmark_duration_seconds(
        SteadyClock<>::type::time_point start,
        SteadyClock<>::type::time_point end) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

static SteadyClock<>::type::time_point kStart;

ScopeTimer::ScopeTimer(const char *name) {
    mFinished = true;

    reset(name);
}

ScopeTimer::~ScopeTimer() {
    if (!mFinished) {
        finish();
    }
}

void ScopeTimer::finish() {
    mFinished = true;
    double best = std::numeric_limits<double>::infinity();
    auto end = benchmark_now();
    double elapsed_seconds = benchmark_duration_seconds(kStart, end);
    best = std::min(best, elapsed_seconds);
    printf("leave %s %.3fms\n", mName.c_str(), best*1000);
}

void ScopeTimer::reset(const char *name) {
    if (!mFinished) {
        finish();
    }
    mName = name;
    mFinished = false;
    kStart = benchmark_now();
    printf("enter %s\n", mName.c_str());
}
