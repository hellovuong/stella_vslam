//
// Created by vuong on 3/6/23.
//

#include "sp_extractor.h"

namespace stella_vslam::feature {
sp_extractor::sp_extractor(sp_params super_point_config)
    : sp_params_(std::move(super_point_config)),
      sp_ptr_(std::make_shared<sp_trt>(sp_params_)) {
    feature_type_ = feature_type_t::SuperPoint;
    image_pyramid_.resize(num_levels_);
    num_feature_per_level_.resize(num_levels_);

    auto distribute_factor = 1 / scale_factor_;
    compute_num_features_per_level(distribute_factor, num_levels_,
                                   max_num_features_, num_feature_per_level_);
    scale_factors_ = orb_params::calc_scale_factors(num_levels_, scale_factor_);
    inv_scale_factors_ = orb_params::calc_inv_scale_factors(num_levels_, scale_factor_);
    level_sigma_sq_ = orb_params::calc_level_sigma_sq(num_levels_, scale_factor_);
    inv_level_sigma_sq_ = orb_params::calc_inv_level_sigma_sq(num_levels_, scale_factor_);
}
sp_extractor::sp_extractor(const YAML::Node& yaml_node) {
    YAML::Node superpoint_node = yaml_node["Feature"];

    parse_configs(superpoint_node);

    feature_type_ = feature_type_t::SuperPoint;
    num_levels_ = superpoint_node["num_levels"].as<unsigned int>(8);
    scale_factor_ = superpoint_node["scale_factor"].as<float>(1.2);

    sp_ptr_ = std::make_shared<sp_trt>(sp_params_);

    image_pyramid_.resize(num_levels_);
    num_feature_per_level_.resize(num_levels_);

    auto distribute_factor = 1 / scale_factor_;
    compute_num_features_per_level(distribute_factor, num_levels_,
                                   max_num_features_, num_feature_per_level_);
    scale_factors_ = orb_params::calc_scale_factors(num_levels_, scale_factor_);
    inv_scale_factors_ = orb_params::calc_inv_scale_factors(num_levels_, scale_factor_);
    level_sigma_sq_ = orb_params::calc_level_sigma_sq(num_levels_, scale_factor_);
    inv_level_sigma_sq_ = orb_params::calc_inv_level_sigma_sq(num_levels_, scale_factor_);
}
void sp_extractor::parse_configs(const YAML::Node& yaml_node) {
    max_num_features_ = yaml_node["max_keypoints"].as<int>();
    sp_params_.max_keypoints = (int)max_num_features_;

    sp_params_.keypoint_threshold = yaml_node["keypoint_threshold"].as<double>();
    sp_params_.remove_borders = yaml_node["remove_borders"].as<int>();
    sp_params_.dla_core = yaml_node["dla_core"].as<int>();

    YAML::Node superpoint_input_tensor_names_node = yaml_node["input_tensor_names"];
    size_t superpoint_num_input_tensor_names = superpoint_input_tensor_names_node.size();
    for (size_t i = 0; i < superpoint_num_input_tensor_names; i++) {
        sp_params_.input_tensor_names.push_back(superpoint_input_tensor_names_node[i].as<std::string>());
    }

    YAML::Node superpoint_output_tensor_names_node = yaml_node["output_tensor_names"];
    size_t superpoint_num_output_tensor_names = superpoint_output_tensor_names_node.size();
    for (size_t i = 0; i < superpoint_num_output_tensor_names; i++) {
        sp_params_.output_tensor_names.push_back(superpoint_output_tensor_names_node[i].as<std::string>());
    }

    auto superpoint_onnx_file = yaml_node["onnx_file"].as<std::string>();
    auto superpoint_engine_file = yaml_node["engine_file"].as<std::string>();
    sp_params_.onnx_file = superpoint_onnx_file;
    sp_params_.engine_file = superpoint_engine_file;
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

    for (int level = 0; level < (int)num_levels_; ++level) {
        num_keypoints += (int)all_keypoints[level].size();
        for (auto keypoint : all_keypoints[level]) {
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