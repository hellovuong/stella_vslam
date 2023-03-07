//
// Created by vuong on 3/6/23.
//

#include "sp_extractor.h"
#include "orb_params.h"

namespace stella_vslam::feature {
sp_extractor::sp_extractor() {
    image_pyramid_.resize(num_levels_);
    num_feature_per_level_.resize(num_levels_);

    auto distribute_factor = 1 / scale_factor_;

    compute_num_features_per_level(distribute_factor, num_levels_, max_num_features_, num_feature_per_level_);

    scale_factors_ = orb_params::calc_scale_factors(num_levels_, scale_factor_);
    inv_scale_factors_ = orb_params::calc_inv_scale_factors(num_levels_, scale_factor_);
    level_sigma_sq_ = orb_params::calc_level_sigma_sq(num_levels_, scale_factor_);
    inv_level_sigma_sq_ = orb_params::calc_inv_level_sigma_sq(num_levels_, scale_factor_);
}
void sp_extractor::extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                           std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors) {
    if (in_image.empty()) {
        return;
    }
    // get cv::Mat of image
    const auto image = in_image.getMat();
    compute_image_pyramid(image, num_levels_, image_pyramid_, scale_factors_);

    int num_keypoints = 0;
    std::vector<std::vector<cv::KeyPoint>> all_keypoints(num_levels_);
    std::vector<cv::Mat> all_descriptors(num_levels_);

    // exact multi layer parallel
    DetectParallel detector(all_keypoints.data(), all_descriptors.data(), this);
    cv::parallel_for_(cv::Range(0, (int)num_levels_), detector);

    for (int level = 0; level < num_levels_; ++level)
    {
        for (auto keypoint : all_keypoints[level])
        {
            keypoint.octave = level;
            keypoint.pt *= scale_factors_[level];
            keypts.emplace_back(keypoint);
        }
    }
    cv::vconcat(all_descriptors.data(), all_descriptors.size(), out_descriptors);
}

void sp_extractor::compute_num_features_per_level(const float distributed_factor, size_t num_levels,
                                                  const size_t max_num_features,
                                                  std::vector<size_t>& num_feature_per_level) const {
    auto desired_per_level = (float)max_num_features_ * (1 - distributed_factor) / (1 - (float)pow((double)distributed_factor, (double)num_levels));
    int sum_features = 0;
    for (size_t level = 0; level < num_levels - 1; level++) {
        num_feature_per_level[level] = cvRound(desired_per_level);
        sum_features += (int)num_feature_per_level[level];
        desired_per_level *= distributed_factor;
    }
    num_feature_per_level[num_levels - 1] = std::max((int)max_num_features - sum_features, 0);
}
void sp_extractor::compute_image_pyramid(const cv::Mat& image, const size_t num_levels,
                                         std::vector<cv::Mat>& image_pyramid, std::vector<float>& scale_factors) {
    image_pyramid.at(0) = image;
    for (unsigned int level = 1; level < num_levels; ++level) {
        // determine the size of an image
        const double scale = scale_factors.at(level);
        const cv::Size size((int)std::round(image.cols * 1.0 / scale), (int)std::round(image.rows * 1.0 / scale));
        // resize
        cv::resize(image_pyramid.at(level - 1), image_pyramid.at(level), size, 0, 0, cv::INTER_LINEAR);
    }
}
} // namespace stella_vslam::feature