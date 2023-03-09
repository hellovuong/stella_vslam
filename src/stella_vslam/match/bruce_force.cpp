//
// Created by vuong on 3/9/23.
//

#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/frame.h"
#include "stella_vslam/data/landmark.h"

#include "stella_vslam/match/bruce_force.h"

namespace stella_vslam::match {
size_t bruce_force::match(const std::shared_ptr<data::keyframe>& keyfrm, data::frame& frm,
                          std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm) {
    cv::BFMatcher matcher;

    if (frm.frm_obs_.descriptors_.type() == CV_32F)
        matcher = cv::BFMatcher(cv::NORM_L2, true);
    else
        matcher = cv::BFMatcher(cv::NORM_HAMMING);

    unsigned int num_matches = 0;
    matched_lms_in_frm = std::vector<std::shared_ptr<data::landmark>>(frm.frm_obs_.num_keypts_, nullptr);
    const auto keyfrm_lms = keyfrm->get_landmarks();

    std::vector<int> vRealIndexKF;
    vRealIndexKF.reserve(keyfrm_lms.size());
    for (int realIdxKF = 0; realIdxKF < keyfrm->frm_obs_.descriptors_.rows; ++realIdxKF) {
        auto pMP = keyfrm->get_landmarks()[realIdxKF];
        if (!pMP or pMP->will_be_erased())
            continue;
        vRealIndexKF.emplace_back(realIdxKF);
    }

    cv::Mat realDescriptorsKF = cv::Mat(vRealIndexKF.size(), keyfrm->frm_obs_.descriptors_.cols, keyfrm->frm_obs_.descriptors_.type());
    for (size_t index = 0; index < vRealIndexKF.size(); ++index)
        keyfrm->frm_obs_.descriptors_.row(vRealIndexKF[index]).copyTo(realDescriptorsKF.row((int)index));

    std::vector<cv::DMatch> matches;
    matcher.match(realDescriptorsKF, frm.frm_obs_.descriptors_, matches);

    for (auto& match : matches) {
        if (match.distance < TH_LOW) {
            num_matches++;
            int realIdxKF = vRealIndexKF[match.queryIdx];
            int bestIdxF = match.trainIdx;
            matched_lms_in_frm[bestIdxF] = keyfrm_lms[realIdxKF];
        }
    }
    return num_matches;
}
} // namespace stella_vslam::match