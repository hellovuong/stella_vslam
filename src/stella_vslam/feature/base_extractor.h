//
// Created by vuong on 3/7/23.
//

#ifndef STELLA_VSLAM_BASE_FEATURE_H
#define STELLA_VSLAM_BASE_FEATURE_H
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

namespace stella_vslam::feature {
class base_extractor {
public:
    //! Extract keypoints and each descriptor of them
    /**
     *
     * @param in_image
     * @param in_image_mask
     * @param keypts
     * @param out_descriptors
     */
    virtual void extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                         std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors)
        = 0;

    //! Image pyramid
    std::vector<cv::Mat> image_pyramid_;

protected:
    //! mask for extract
    cv::Mat rect_mask_;
};
} // namespace stella_vslam::feature
#endif // STELLA_VSLAM_BASE_FEATURE_H
