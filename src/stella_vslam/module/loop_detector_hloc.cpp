//
// Created by vuong on 2/28/23.
//

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "loop_detector_hloc.h"

#define DEBUG_IMAGE true

namespace stella_vslam::module {
void loop_detector_hloc::enable_loop_detector() {
    loop_detector_is_enabled_ = true;
}
void loop_detector_hloc::disable_loop_detector() {
    loop_detector_is_enabled_ = false;
}
bool loop_detector_hloc::is_enabled() const {
    return loop_detector_is_enabled_;
}
void loop_detector_hloc::set_current_keyframe(const std::shared_ptr<data::keyframe>& keyfrm) {
    cur_keyfrm_ = std::make_shared<hloc::keyframe>(keyfrm);
    detect_loop_candidates();
}
bool loop_detector_hloc::detect_loop_candidates() {
    auto succeeded = detect_loop_candidates_impl();
    return false;
}
bool loop_detector_hloc::detect_loop_candidates_impl() {
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE) {
        auto feature_num = cur_keyfrm_->keyfrm_->get_landmarks().size();
        cv::resize(cur_keyfrm_->keyfrm_->img.clone(), compressed_image, cv::Size(848, 480));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[cur_keyfrm_->keyfrm_->id_] = compressed_image;
    }

    // first query; then add this frame into database!
    vector<Result> ret;
    cv::Mat global_desp = cur_keyfrm_->global_descriptors_.clone();

    if (last_loop_count > 10) {
        db.query(global_desp, ret);
        db.add(global_desp);
    }
    else {
        db.add(global_desp);
        last_loop_count++;
    }


    bool find_loop = false;
    cv::Mat loop_result;
    if (DEBUG_IMAGE) {
        loop_result = compressed_image.clone();
        if (ret.size() > 0)
            putText(loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    }
    // visual loop result
    if (DEBUG_IMAGE) {
        if (ret.size() >= 1 && ret[0].Score > LOOP_THRESHOLD) {
            int tmp_index = ret[0].Id;
            auto it = image_pool.find(tmp_index);
            cv::Mat tmp_image = (it->second).clone();
            if (not tmp_image.empty()) {
                putText(tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(ret[0].Score), cv::Point2f(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
                cv::hconcat(loop_result, tmp_image, loop_result);
            }
            last_loop_count = 0;
        }
    }
    if(DEBUG_IMAGE) {
        cv::imshow("loop_result", loop_result);
        cv::waitKey(100);
    }

    return false;
}

} // namespace stella_vslam::module