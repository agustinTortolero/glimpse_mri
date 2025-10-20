#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace imgutil {

// --- NEW ---
// Robust float* [H x W] -> 8-bit converter (NaN/Inf-safe) with debug prints.
// Matches the behavior you had inside AppController.
inline cv::Mat to_u8_slice(const float* src, int H, int W)
{
    if (!src || H <= 0 || W <= 0) {
        std::cerr << "[DBG][imgutil] to_u8_slice: null ptr or bad size H=" << H << " W=" << W << "\n";
        return {};
    }

    const size_t HW = static_cast<size_t>(H) * static_cast<size_t>(W);
    double mn = +std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < HW; ++i) {
        const double v = static_cast<double>(src[i]);
        if (!std::isfinite(v)) continue;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }

    if (!std::isfinite(mn) || !std::isfinite(mx) || mx <= mn) {
        std::cerr << "[DBG][imgutil] to_u8_slice: invalid min/max (mn=" << mn << " mx=" << mx << "); returning zeros\n";
        return cv::Mat::zeros(H, W, CV_8UC1);
    }

    std::cerr << "[DBG][imgutil] to_u8_slice: min=" << mn << " max=" << mx << "\n";

    cv::Mat out(H, W, CV_8UC1);
    const double scale = 255.0 / (mx - mn);
    for (int y = 0; y < H; ++y) {
        const float* srow = src + static_cast<size_t>(y) * W;
        uint8_t* drow = out.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            double v = static_cast<double>(srow[x]);
            if (!std::isfinite(v)) v = mn;
            const int u = static_cast<int>((v - mn) * scale + 0.5);
            drow[x] = static_cast<uint8_t>(std::clamp(u, 0, 255));
        }
    }
    return out;
}

// Existing: vector<float> -> 8U with OpenCV normalize (kept as-is)
inline cv::Mat to_8u(const std::vector<float>& data, int h, int w)
{
    if (data.empty() || h <= 0 || w <= 0) {
        std::cerr << "[DBG][imgutil] to_8u: empty input or bad size h=" << h << " w=" << w << "\n";
        return {};
    }

    const size_t need = (size_t)h * w;
    if (need != data.size()) {
        std::cerr << "[ERR][imgutil] to_8u: size mismatch vec=" << data.size()
        << " vs HxW=" << h << "x" << w << " => " << need << "\n";
        return {};
    }

    cv::Mat f(h, w, CV_32F, const_cast<float*>(data.data()));
    double mn = 0.0, mx = 0.0;
    cv::minMaxLoc(f, &mn, &mx);
    std::cerr << "[DBG][imgutil] to_8u: min=" << mn << " max=" << mx << "\n";

    if (!std::isfinite(mn) || !std::isfinite(mx)) {
        std::cerr << "[DBG][imgutil] to_8u: non-finite min/max; returning zeros\n";
        return cv::Mat::zeros(h, w, CV_8U);
    }
    if (std::abs(mx - mn) < 1e-20) {
        std::cerr << "[DBG][imgutil] to_8u: flat image (min==max); returning zeros\n";
        return cv::Mat::zeros(h, w, CV_8U);
    }

    cv::Mat out8;
    cv::normalize(f, out8, 0, 255, cv::NORM_MINMAX, CV_8U);
    return out8;
}

// Simple left-to-right gradient for sanity checks.
inline cv::Mat make_test_gradient(int h, int w)
{
    if (h <= 0) h = 256;
    if (w <= 0) w = 256;
    cv::Mat g(h, w, CV_8U);
    for (int y = 0; y < h; ++y) {
        auto row = g.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            row[x] = static_cast<uint8_t>((255.0 * x) / std::max(1, w - 1));
        }
    }
    return g;
}

} // namespace imgutil
