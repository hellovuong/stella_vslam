//
// Created by vuong on 3/6/23.
//

#ifndef STELLA_VSLAM_SP_EXTRACTOR_H
#define STELLA_VSLAM_SP_EXTRACTOR_H

#include "stella_vslam/feature/base_extractor.h"
//#include "stella_vslam/hloc/hloc.h"
#include "stella_vslam/feature/orb_params.h"
#include "stella_vslam/feature/sp_trt.h"

namespace stella_vslam::feature {

class sp_extractor : public base_extractor {
public:
    explicit sp_extractor(sp_params super_point_config);
    explicit sp_extractor(const YAML::Node& yaml_node);

    ~sp_extractor() override = default;

    //! Extract keypoints and each descriptor of them
    void extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                 std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors) override;

    void parse_configs(const YAML::Node& yaml_node);

    sp_params sp_params_;
    spPtr sp_ptr_;

    std::vector<size_t> num_feature_per_level_ = {};

private:
    /**
     *
     * @param distributed_factor
     * @param num_levels
     * @param max_num_features
     * @param num_feature_per_level [out]
     */
    void compute_num_features_per_level(float distributed_factor, size_t num_levels, size_t max_num_features,
                                        std::vector<size_t>& num_feature_per_level) const;

    static void compute_image_pyramid(const cv::Mat& image, size_t num_levels, std::vector<cv::Mat>& image_pyramid,
                                      std::vector<float>& scale_factors);

    unsigned int num_levels_ = 4;
    size_t max_num_features_ = 1000;
    float scale_factor_ = 1.2;

    //! A list of the scale factor of each pyramid layer
    std::vector<float> scale_factors_;
    std::vector<float> inv_scale_factors_;
    //! A list of the sigma of each pyramid layer
    std::vector<float> level_sigma_sq_;
    std::vector<float> inv_level_sigma_sq_;
};
class DetectParallel : public cv::ParallelLoopBody {
public:
    DetectParallel(std::vector<cv::KeyPoint>* all_keypoints, cv::Mat* all_descriptors, sp_extractor* extractor)
        : all_keypoints_(all_keypoints), all_descriptors_(all_descriptors), extractor_(extractor) {}

    void operator()(const cv::Range& range) const CV_OVERRIDE {
        for (int level = range.start; level != range.end; ++level) {
            extractor_->sp_ptr_->detect_and_compute(extractor_->image_pyramid_.at(level), *all_keypoints_, *all_descriptors_,
                                                    (int)extractor_->num_feature_per_level_.at(level));
        }
    }

    DetectParallel& operator = (const DetectParallel&) {
        return *this;
    };

private:
    std::vector<cv::KeyPoint>* all_keypoints_;

    cv::Mat* all_descriptors_;
    sp_extractor* extractor_;
};
} // namespace stella_vslam::feature

#endif // STELLA_VSLAM_SP_EXTRACTOR_H
