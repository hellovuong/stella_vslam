//
// Created by vuong on 3/7/23.
//

#ifndef STELLA_VSLAM_BASE_FEATURE_H
#define STELLA_VSLAM_BASE_FEATURE_H
#include <opencv2/core/mat.hpp>
namespace stella_vslam::feature {
class base_feature {
    //! Extract keypoints and each descriptor of them
    /**
     *
     * @param in_image
     * @param in_image_mask
     * @param keypts
     * @param out_descriptors
     */
    virtual void extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                         std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors) const
        = 0;
};
} // namespace stella_vslam::feature
#endif // STELLA_VSLAM_BASE_FEATURE_H
