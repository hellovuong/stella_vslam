#ifndef STELLA_VSLAM_MATCH_BASE_H
#define STELLA_VSLAM_MATCH_BASE_H

#include <array>
#include <algorithm>
#include <numeric>

#include <opencv2/core/mat.hpp>
#include <stella_vslam/type.h>

namespace stella_vslam::match {

static constexpr float HAMMING_DIST_THR_LOW = 50;
static constexpr float HAMMING_DIST_THR_HIGH = 100;
static constexpr float MAX_HAMMING_DIST = 256;

static constexpr float HAMMING_L2_DIST_THR_HIGH = 0.75;
static constexpr float HAMMING_L2_DIST_THR_LOW = 0.6;
static constexpr float MAX_HAMMING_L2_DIST = 1.0f;

enum class descriptor_type {
    BIN,
    FLOAT
};
typedef stella_vslam::match::descriptor_type descriptor_type_t;

//! ORB特徴量間のハミング距離を計算する
inline unsigned int compute_descriptor_distance_32(const cv::Mat& desc_1, const cv::Mat& desc_2) {
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    assert(desc_1.type() == CV_8U);
    assert(desc_2.type() == CV_8U);

    constexpr uint32_t mask_1 = 0x55555555U;
    constexpr uint32_t mask_2 = 0x33333333U;
    constexpr uint32_t mask_3 = 0x0F0F0F0FU;
    constexpr uint32_t mask_4 = 0x01010101U;

    const auto* pa = desc_1.ptr<uint32_t>();
    const auto* pb = desc_2.ptr<uint32_t>();

    unsigned int dist = 0;

    for (unsigned int i = 0; i < 8; ++i, ++pa, ++pb) {
        auto v = *pa ^ *pb;
        v -= ((v >> 1) & mask_1);
        v = (v & mask_2) + ((v >> 2) & mask_2);
        dist += (((v + (v >> 4)) & mask_3) * mask_4) >> 24;
    }

    return dist;
}

//! ORB特徴量間のハミング距離を計算する
inline unsigned int compute_descriptor_distance_64(const cv::Mat& desc_1, const cv::Mat& desc_2) {
    // https://stackoverflow.com/questions/21826292/t-sql-hamming-distance-function-capable-of-decimal-string-uint64?lq=1

    constexpr uint64_t mask_1 = 0x5555555555555555UL;
    constexpr uint64_t mask_2 = 0x3333333333333333UL;
    constexpr uint64_t mask_3 = 0x0F0F0F0F0F0F0F0FUL;
    constexpr uint64_t mask_4 = 0x0101010101010101UL;

    const auto* pa = desc_1.ptr<uint64_t>();
    const auto* pb = desc_2.ptr<uint64_t>();

    unsigned int dist = 0;

    for (unsigned int i = 0; i < 4; ++i, ++pa, ++pb) {
        auto v = *pa ^ *pb;
        v -= (v >> 1) & mask_1;
        v = (v & mask_2) + ((v >> 2) & mask_2);
        dist += (((v + (v >> 4)) & mask_3) * mask_4) >> 56;
    }

    return dist;
}

inline float compute_descriptor_distance_l2(const cv::Mat& a, const cv::Mat& b) {
    // Eigen is much faster than OpenCV
    assert(a.cols == b.cols);
    assert(a.isContinuous() && b.isContinuous());
    assert(a.type() == CV_32F);
    assert(b.type() == CV_32F);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> des1(a.ptr<float>(), a.rows, a.cols);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> des2(b.ptr<float>(), b.rows, b.cols);
    return (des1 - des2).norm();
}

class base {
public:
    base(const float lowe_ratio, const bool check_orientation)
        : lowe_ratio_(lowe_ratio), check_orientation_(check_orientation) {}

    virtual ~base() = default;

    descriptor_type_t descriptor_type_ = descriptor_type_t::BIN;

protected:
    const float lowe_ratio_;
    const bool check_orientation_;
};

} // namespace stella_vslam

#endif // STELLA_VSLAM_MATCH_BASE_H
