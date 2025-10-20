
#pragma once
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace imgutil {

// Debuggy float->8U with min/max prints and flat-image guard.
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

