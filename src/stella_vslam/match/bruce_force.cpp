//
// Created by vuong on 3/9/23.
//

#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/frame.h"
#include "stella_vslam/data/landmark.h"

#include "stella_vslam/match/bruce_force.h"

namespace stella_vslam::match {
size_t bruce_force::match(const std::shared_ptr<data::keyframe>& keyfrm, data::frame& frm,
                          std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm,
                          std::vector<cv::DMatch>& best_matches) {
    cv::BFMatcher matcher;

    if (frm.frm_obs_.descriptors_.type() == CV_32F)
        matcher = cv::BFMatcher(cv::NORM_L2, true);
    else
        matcher = cv::BFMatcher(cv::NORM_HAMMING);

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
    best_matches.clear();
    for (auto& match : matches) {
        if (match.distance < TH_LOW) {
            int realIdxKF = vRealIndexKF[match.queryIdx];
            int bestIdxF = match.trainIdx;
            matched_lms_in_frm[bestIdxF] = keyfrm_lms[realIdxKF];
            best_matches.push_back(match);
        }
    }
    return best_matches.size();
}
size_t bruce_force::match(const std::shared_ptr<data::keyframe>& keyfrm, data::frame& frm,
                          std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm) {
    cv::BFMatcher matcher;
    size_t num_matches = 0;
    if (frm.frm_obs_.descriptors_.type() == CV_32F)
        matcher = cv::BFMatcher(cv::NORM_L2, true);
    else
        matcher = cv::BFMatcher(cv::NORM_HAMMING);

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
    for (size_t index = 0; index < vRealIndexKF.size(); ++index) {
        keyfrm->frm_obs_.descriptors_.row(vRealIndexKF[index]).copyTo(realDescriptorsKF.row((int)index));
    }
    std::vector<cv::DMatch> matches;
    matcher.match(realDescriptorsKF, frm.frm_obs_.descriptors_, matches);
    for (auto& match : matches) {
        if (match.distance < TH_LOW) {
            int realIdxKF = vRealIndexKF[match.queryIdx];
            int bestIdxF = match.trainIdx;
            matched_lms_in_frm[bestIdxF] = keyfrm_lms[realIdxKF];
            num_matches++;
        }
    }
    return num_matches;
}
sg_matcher::sg_matcher(const YAML::Node& node) {
    spdlog::info("CONSTRUCT: SuperGlue Matcher");
    SuperGlueConfig config;
    SuperGlue::gen_configs(node, config);
    superGlue = std::make_unique<SuperGlue>(config);
}

size_t sg_matcher::match(const std::shared_ptr<data::keyframe>& keyfrm, data::frame& frm,
                         std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm,
                         std::vector<cv::DMatch>& matches) {
    matches.clear();
    matched_lms_in_frm = std::vector<std::shared_ptr<data::landmark>>(frm.frm_obs_.num_keypts_, nullptr);
    const auto& keyfrm_lms = keyfrm->get_landmarks();

    superGlue->MatchingPoints(keyfrm->frm_obs_.undist_keypts_, keyfrm->frm_obs_.descriptors_,
                              frm.frm_obs_.undist_keypts_, frm.frm_obs_.descriptors_, matches);

    // Perform homography estimation using RANSAC
    cv::Mat inliersMask;
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (auto match : matches) {
        pts1.emplace_back(keyfrm->frm_obs_.undist_keypts_[match.queryIdx].pt);
        pts2.emplace_back(frm.frm_obs_.undist_keypts_[match.trainIdx].pt);
    }

    if (pts1.empty() or pts2.empty())
        return 0;

    cv::findHomography(pts1, pts2, cv::RANSAC, 2, inliersMask);

    // Filter matches based on inliers mask
    std::vector<cv::Point2f> filteredSrcPoints, filteredDstPoints;
    for (int i = 0; i < inliersMask.rows; ++i) {
        if (not inliersMask.at<uchar>(i)) {
            matches[i].distance = -1;
        }
    }
    int count = 0;
    for (auto& match : matches) {
        if (match.distance < 0)
            continue;
        const auto& lm = keyfrm_lms.at(match.queryIdx);
        if (!lm or lm->will_be_erased()) {
            match.distance = -1;
            continue;
        }
        int bestIdxF = match.trainIdx;
        matched_lms_in_frm[bestIdxF] = lm;
        count++;
    }
    return count;
}
void sg_matcher::drawMatches(const cv::Mat& img1, const cv::Mat& img2,
                             const std::vector<cv::KeyPoint>& undist_keypts_1, const std::vector<cv::KeyPoint>& undist_keypts_2,
                             const std::vector<cv::DMatch>& matches_result) {
    cv::Mat show_img;
    cv::drawMatches(img1, undist_keypts_1, img2, undist_keypts_2, matches_result, show_img);
    cv::imwrite("matches_w_candidates.png", show_img);
}

size_t sg_matcher::match_current_and_last_frames(data::frame& curr_frm, const data::frame& last_frm,
                                                 std::vector<cv::DMatch>& best_matches) const {
    auto matched_lms_in_curr_frm = std::vector<std::shared_ptr<data::landmark>>(curr_frm.frm_obs_.num_keypts_, nullptr);
    const auto lms_in_last_frm = last_frm.get_landmarks();
    // 0 - queryIdx, 1 - trainIdx
    superGlue->MatchingPoints(last_frm.frm_obs_.undist_keypts_, last_frm.frm_obs_.descriptors_,
                              curr_frm.frm_obs_.undist_keypts_, curr_frm.frm_obs_.descriptors_, best_matches);
    // Perform homography estimation using RANSAC
    cv::Mat inliersMask;
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (auto match : best_matches) {
        pts1.emplace_back(last_frm.frm_obs_.undist_keypts_[match.queryIdx].pt);
        pts2.emplace_back(curr_frm.frm_obs_.undist_keypts_[match.trainIdx].pt);
    }

    if (pts1.empty() or pts2.empty())
        return 0;

    cv::findHomography(pts1, pts2, cv::RANSAC, 2, inliersMask);

    // Filter matches based on inliers mask
    std::vector<cv::Point2f> filteredSrcPoints, filteredDstPoints;
    for (int i = 0; i < inliersMask.rows; ++i) {
        if (not inliersMask.at<uchar>(i)) {
            best_matches[i].distance = -1;
        }
    }
    for (auto& match : best_matches) {
        int bestIdxF = match.trainIdx;
        if (!lms_in_last_frm[match.queryIdx] or lms_in_last_frm[match.queryIdx]->will_be_erased())
            continue;
        curr_frm.add_landmark(lms_in_last_frm[match.queryIdx], bestIdxF);
    }

    return best_matches.size();
}
} // namespace stella_vslam::match
