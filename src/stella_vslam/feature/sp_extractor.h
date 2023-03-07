//
// Created by vuong on 3/6/23.
//

#ifndef STELLA_VSLAM_SP_EXTRACTOR_H
#define STELLA_VSLAM_SP_EXTRACTOR_H

#include "stella_vslam/feature/base_extractor.h"
#include "stella_vslam/hloc/hloc.h"

namespace stella_vslam::feature {

class sp_extractor : public base_extractor {
public:
    sp_extractor();
    ~sp_extractor() = default;

    //! Extract keypoints and each descriptor of them
    void extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                 std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors) override;

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

    const unsigned int num_levels_ = 4;
    size_t max_num_features_ = 1000;
    const float scale_factor_ = 1.2;
    std::vector<size_t> num_feature_per_level_ = {};

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
            stella_vslam::SuperPoint::Extract(extractor_->image_pyramid_.at(level), *all_keypoints_, *all_descriptors_);
        }
    }

    DetectParallel& operator=(const DetectParallel&) {
        return *this;
    };

private:
    std::vector<cv::KeyPoint>* all_keypoints_;

    cv::Mat* all_descriptors_;
    sp_extractor* extractor_;
};
} // namespace stella_vslam::feature

#endif // STELLA_VSLAM_SP_EXTRACTOR_H
