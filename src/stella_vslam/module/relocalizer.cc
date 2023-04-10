#include "stella_vslam/data/frame.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/data/bow_database.h"
#include "stella_vslam/data/hf_net_database.h"
#include "stella_vslam/module/local_map_updater.h"
#include "stella_vslam/module/relocalizer.h"
#include "stella_vslam/optimize/pose_optimizer_g2o.h"
#include "stella_vslam/util/fancy_index.h"

#include <spdlog/spdlog.h>

#include <memory>
#include <utility>

namespace stella_vslam::module {

relocalizer::relocalizer(std::shared_ptr<optimize::pose_optimizer> pose_optimizer,
                         const double bow_match_lowe_ratio, const double proj_match_lowe_ratio,
                         const double robust_match_lowe_ratio,
                         const unsigned int min_num_bow_matches, const unsigned int min_num_valid_obs,
                         const bool use_fixed_seed,
                         const YAML::Node& sg_yaml_node)
    : min_num_bow_matches_(min_num_bow_matches), min_num_valid_obs_(min_num_valid_obs),
      bow_matcher_((float)bow_match_lowe_ratio, false), proj_matcher_((float)proj_match_lowe_ratio, false),
      robust_matcher_((float)robust_match_lowe_ratio, false),
      sg_matcher_(std::make_unique<match::sg_matcher>(sg_yaml_node)),
      pose_optimizer_(std::move(pose_optimizer)), use_fixed_seed_(use_fixed_seed) {
    spdlog::debug("CONSTRUCT: module::relocalizer");
}

relocalizer::relocalizer(const std::shared_ptr<optimize::pose_optimizer>& pose_optimizer, const YAML::Node& yaml_node)
    : relocalizer(pose_optimizer,
                  yaml_node["bow_match_lowe_ratio"].as<double>(0.75),
                  yaml_node["proj_match_lowe_ratio"].as<double>(0.9),
                  yaml_node["robust_match_lowe_ratio"].as<double>(0.8),
                  yaml_node["min_num_bow_matches"].as<unsigned int>(20),
                  yaml_node["min_num_valid_obs"].as<unsigned int>(50),
                  yaml_node["use_fixed_seed"].as<bool>(false),
                  yaml_node["SuperGlue"]) {
}

relocalizer::~relocalizer() {
    spdlog::debug("DESTRUCT: module::relocalizer");
}

bool relocalizer::relocalize(data::base_place_recognition* vpr_db, data::frame& curr_frm) const {
    // Acquire relocalization candidates
    std::vector<std::shared_ptr<data::keyframe>> reloc_candidates;
    if (vpr_db->database_type == data::place_recognition_t::BoW) {
        reloc_candidates = dynamic_cast<data::bow_database*>(vpr_db)->acquire_keyframes(curr_frm.bow_vec_);
    }
    else if (vpr_db->database_type == data::place_recognition_t::HF_Net) {
        reloc_candidates = dynamic_cast<data::hf_net_database*>(vpr_db)->acquire_keyframes(curr_frm.frm_obs_.global_descriptors_.clone());
    }
    else {
        spdlog::warn("Undefined type of place recognition!");
    }

    if (reloc_candidates.empty()) {
        spdlog::debug("relocalizer::relocalize: Empty reloc candidates");
        return false;
    }

    return reloc_by_candidates(curr_frm, reloc_candidates);
}

bool relocalizer::reloc_by_candidates(data::frame& curr_frm,
                                      const std::vector<std::shared_ptr<stella_vslam::data::keyframe>>& reloc_candidates,
                                      bool use_robust_matcher) const {
    const auto num_candidates = reloc_candidates.size();

    spdlog::debug("relocalizer::reloc_by_candidates: Start relocalization. Number of candidate keyframes is {}", num_candidates);

    // Compute matching points for each candidate by using BoW tree matcher
    for (unsigned int i = 0; i < num_candidates; ++i) {
        const auto& candidate_keyfrm = reloc_candidates.at(i);
        if (candidate_keyfrm->will_be_erased()) {
            spdlog::debug("keyframe will be erased. candidate keyframe id is {}", candidate_keyfrm->id_);
            continue;
        }

        bool ok = reloc_by_candidate(curr_frm, candidate_keyfrm, use_robust_matcher);
        if (ok) {
            spdlog::info("relocalization succeeded (id={})", candidate_keyfrm->id_);
            // TODO: should set the reference keyframe of the current frame
            return true;
        }
    }

    curr_frm.invalidate_pose();
    return false;
}

bool relocalizer::reloc_by_candidate(data::frame& curr_frm,
                                     const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                                     bool use_robust_matcher) const {
    std::vector<unsigned int> inlier_indices;
    std::vector<std::shared_ptr<data::landmark>> matched_landmarks;
    bool ok = relocalize_by_pnp_solver(curr_frm, candidate_keyfrm, use_robust_matcher, inlier_indices, matched_landmarks);
    if (!ok) {
        return false;
    }

    // Set 2D-3D matches for the pose optimization
    curr_frm.erase_landmarks();
    for (const auto idx : inlier_indices) {
        // Set only the valid 3D points to the current frame
        curr_frm.add_landmark(matched_landmarks.at(idx), idx);
    }
    spdlog::debug("2. optimize_pose");
    std::vector<bool> outlier_flags;
    ok = optimize_pose(curr_frm, candidate_keyfrm, outlier_flags);
    if (!ok) {
        return false;
    }

    std::set<std::shared_ptr<data::landmark>> already_found_landmarks;
    for (const auto idx : inlier_indices) {
        if (outlier_flags.at(idx)) {
            continue;
        }
        // Record the 3D points already associated to the frame keypoints
        already_found_landmarks.insert(matched_landmarks.at(idx));
    }

    ok = refine_pose(curr_frm, candidate_keyfrm, already_found_landmarks);
    if (!ok) {
        return false;
    }

    ok = refine_pose_by_local_map(curr_frm, candidate_keyfrm);
    return ok;
}

bool relocalizer::relocalize_by_pnp_solver(data::frame& curr_frm,
                                           const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                                           bool use_robust_matcher,
                                           std::vector<unsigned int>& inlier_indices,
                                           std::vector<std::shared_ptr<data::landmark>>& matched_landmarks) const {
    size_t num_matches;
    std::vector<cv::DMatch> match_result;
    if (curr_frm.frm_obs_.descriptors_.type() == CV_8U) {
        num_matches = use_robust_matcher ? robust_matcher_.match_frame_and_keyframe(curr_frm, candidate_keyfrm, matched_landmarks)
                                         : bow_matcher_.match_frame_and_keyframe(candidate_keyfrm, curr_frm, matched_landmarks);
    }
    else {
        num_matches = sg_matcher_->match(candidate_keyfrm, curr_frm, matched_landmarks, match_result);
    }
    // Discard the candidate if the number of 2D-3D matches is less than the threshold
    if (num_matches < min_num_bow_matches_) {
        return false;
    }

    drawMatches(curr_img_, candidate_keyfrm->img.clone(), curr_frm.frm_obs_.undist_keypts_, candidate_keyfrm->frm_obs_.undist_keypts_, match_result);

    // Set up an PnP solver with the current 2D-3D matches
    const auto valid_indices = extract_valid_indices(matched_landmarks);
    auto pnp_solver = setup_pnp_solver(valid_indices, curr_frm.frm_obs_.bearings_, curr_frm.frm_obs_.undist_keypts_,
                                       matched_landmarks, curr_frm.orb_params_->scale_factors_);

    // 1. Estimate the camera pose using EPnP (+ RANSAC)
    spdlog::debug("1. Estimate the camera pose using EPnP (+ RANSAC)");
    spdlog::debug("- valid / total : {} / {}", valid_indices.size(), matched_landmarks.size());
    pnp_solver->find_via_ransac(30, false);
    if (!pnp_solver->solution_is_valid()) {
        spdlog::debug("solution is not valid. candidate keyframe id is {}", candidate_keyfrm->id_);
        return false;
    }

    curr_frm.set_pose_cw(pnp_solver->get_best_cam_pose());

    // Get the inlier indices after EPnP+RANSAC
    inlier_indices = util::resample_by_indices(valid_indices, pnp_solver->get_inlier_flags());

    return true;
}

bool relocalizer::optimize_pose(data::frame& curr_frm,
                                const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                                std::vector<bool>& outlier_flags) const {
    // Pose optimization
    Mat44_t optimized_pose;
    auto num_valid_obs = pose_optimizer_->optimize(curr_frm, optimized_pose, outlier_flags);
    curr_frm.set_pose_cw(optimized_pose);

    // Discard the candidate if the number of the inliers is less than the threshold
    if (num_valid_obs < min_num_bow_matches_ / 2) {
        spdlog::debug("relocalizer::optimize_pose: Number of inliers after opt ({}) < threshold ({}). candidate keyframe id is {}",
                      num_valid_obs, min_num_bow_matches_ / 2, candidate_keyfrm->id_);
        return false;
    }

    // Reject outliers
    for (unsigned int idx = 0; idx < curr_frm.frm_obs_.num_keypts_; idx++) {
        if (!outlier_flags.at(idx)) {
            continue;
        }
        curr_frm.erase_landmark_with_index(idx);
    }

    return true;
}

bool relocalizer::refine_pose(data::frame& curr_frm,
                              const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                              const std::set<std::shared_ptr<data::landmark>>& already_found_landmarks) const {
    // 3. Apply projection match to increase 2D-3D matches
    spdlog::debug("3. Apply projection match to increase 2D-3D matches");
    auto num_valid_obs = already_found_landmarks.size();

    // Projection match based on the pre-optimized camera pose
    auto num_found = proj_matcher_.match_frame_and_keyframe(curr_frm, candidate_keyfrm, already_found_landmarks, 10, 100);
    // Discard the candidate if the number of the inliers is less than the threshold
    if (num_valid_obs + num_found < min_num_valid_obs_) {
        spdlog::debug("relocalizer::refine_pose: Number of inliers ({}) after projection match < threshold ({}). candidate keyframe id is {}", num_valid_obs + num_found, min_num_valid_obs_, candidate_keyfrm->id_);
        return false;
    }

    Mat44_t optimized_pose1;
    std::vector<bool> outlier_flags1;
    auto num_valid_obs1 = pose_optimizer_->optimize(curr_frm, optimized_pose1, outlier_flags1);
    spdlog::debug("relocalizer::refine_pose: refine_pose: num_valid_obs1={}", num_valid_obs1);
    curr_frm.set_pose_cw(optimized_pose1);

    // Exclude the already-associated landmarks
    std::set<std::shared_ptr<data::landmark>> already_found_landmarks1;
    for (unsigned int idx = 0; idx < curr_frm.frm_obs_.num_keypts_; ++idx) {
        const auto& lm = curr_frm.get_landmark(idx);
        if (!lm) {
            continue;
        }
        already_found_landmarks1.insert(lm);
    }
    // Apply projection match again, then set the 2D-3D matches
    auto num_additional = proj_matcher_.match_frame_and_keyframe(curr_frm, candidate_keyfrm, already_found_landmarks1, 3, 64);

    // Discard if the number of the observations is less than the threshold
    if (num_valid_obs1 + num_additional < min_num_valid_obs_) {
        spdlog::debug("relocalizer::refine_pose: Number of observations ({}) < threshold ({}). candidate keyframe id is {}",
                      num_valid_obs1 + num_additional, min_num_valid_obs_, candidate_keyfrm->id_);
        return false;
    }

    // Perform optimization again
    Mat44_t optimized_pose2;
    std::vector<bool> outlier_flags2;
    auto num_valid_obs2 = pose_optimizer_->optimize(curr_frm, optimized_pose2, outlier_flags2);
    SPDLOG_TRACE("relocalizer::refine_pose: num_valid_obs2={}", num_valid_obs2);
    curr_frm.set_pose_cw(optimized_pose2);

    // Discard if falling below the threshold
    if (num_valid_obs2 < min_num_valid_obs_) {
        spdlog::debug("relocalizer::refine_pose: Number of observatoins ({}) < threshold ({}). "
            "candidate keyframe id is {}", num_valid_obs2, min_num_valid_obs_, candidate_keyfrm->id_);
        return false;
    }

    // Reject outliers
    for (unsigned int idx = 0; idx < curr_frm.frm_obs_.num_keypts_; ++idx) {
        if (!outlier_flags2.at(idx)) {
            continue;
        }
        curr_frm.erase_landmark_with_index(idx);
    }

    return true;
}

bool relocalizer::refine_pose_by_local_map(data::frame& curr_frm,
                                           const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm) const {
    spdlog::debug("4. relocalizer::refine_pose_by_local_map");
    // Create local map
    constexpr unsigned int max_num_local_keyfrms = 10;
    auto local_map_updater = module::local_map_updater(curr_frm, max_num_local_keyfrms);
    if (!local_map_updater.acquire_local_map()) {
        return false;
    }
    auto local_keyfrms = local_map_updater.get_local_keyframes();
    auto local_landmarks = local_map_updater.get_local_landmarks();
    auto nearest_covisibility = local_map_updater.get_nearest_covisibility();
    SPDLOG_TRACE("refine_pose_by_local_map: keyfrms={}, lms={} nearest_covisibility id={}", local_keyfrms.size(), local_landmarks.size(), nearest_covisibility->id_);

    std::vector<int> margins{5, 15, 5};
    for (size_t i = 0; i < margins.size(); ++i) {
        // select the landmarks which can be reprojected from the ones observed in the current frame
        std::unordered_set<unsigned int> curr_landmark_ids;
        for (const auto& lm : curr_frm.get_landmarks()) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // this landmark cannot be reprojected
            // because already observed in the current frame
            curr_landmark_ids.insert(lm->id_);
        }

        bool found_proj_candidate = false;
        // temporary variables
        Vec2_t reproj;
        float x_right;
        unsigned int pred_scale_level;
        eigen_alloc_unord_map<unsigned int, Vec2_t> lm_to_reproj;
        std::unordered_map<unsigned int, float> lm_to_x_right;
        std::unordered_map<unsigned int, int> lm_to_scale;
        for (const auto& lm : local_landmarks) {
            if (curr_landmark_ids.count(lm->id_)) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // check the observability
            if (curr_frm.can_observe(lm, 0.5, reproj, x_right, pred_scale_level)) {
                lm_to_reproj[lm->id_] = reproj;
                lm_to_x_right[lm->id_] = x_right;
                lm_to_scale[lm->id_] = (int)pred_scale_level;

                found_proj_candidate = true;
            }
        }

        if (!found_proj_candidate) {
            return false;
        }

        // acquire more 2D-3D matches by projecting the local landmarks to the current frame
        match::projection projection_matcher(0.8);
        const auto margin = (float)margins[i];
        auto num_additional_matches = projection_matcher.match_frame_and_landmarks(curr_frm, local_landmarks, lm_to_reproj, lm_to_x_right, lm_to_scale, margin);

        // optimize the pose
        Mat44_t optimized_pose;
        std::vector<bool> outlier_flags;
        auto num_valid_obs = pose_optimizer_->optimize(curr_frm, optimized_pose, outlier_flags);
        curr_frm.set_pose_cw(optimized_pose);

        // Reject outliers
        for (unsigned int idx = 0; idx < curr_frm.frm_obs_.num_keypts_; ++idx) {
            if (!outlier_flags.at(idx)) {
                continue;
            }
            curr_frm.erase_landmark_with_index(idx);
        }
        spdlog::debug("relocalizer::refine_pose_by_local_map: iter={:2}, margin={:2}, num_additional_matches={:4}, num_valid_obs={:4}",
                      i, margin, num_additional_matches, num_valid_obs);

        if (i == margins.size() - 1) {
            const auto num_tracked_lms = candidate_keyfrm->get_num_tracked_landmarks(0);
            const double ratio = 0.1;
            spdlog::debug("relocalizer::refine_pose_by_local_map: num_valid_obs={:4}, num_tracked_lms={:4}", num_valid_obs, num_tracked_lms);
            if (num_valid_obs < num_tracked_lms * ratio) {
                spdlog::debug("relocalizer::refine_pose_by_local_map: Rejected: num_valid_obs={:4} < num_tracked_lms={:4} * 0.2", num_valid_obs, num_tracked_lms);
                return false;
            }
        }
    }

    return true;
}

std::vector<unsigned int> relocalizer::extract_valid_indices(const std::vector<std::shared_ptr<data::landmark>>& landmarks) {
    std::vector<unsigned int> valid_indices;
    valid_indices.reserve(landmarks.size());
    for (unsigned int idx = 0; idx < landmarks.size(); ++idx) {
        const auto& lm = landmarks.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        valid_indices.push_back(idx);
    }
    return valid_indices;
}

std::unique_ptr<solve::pnp_solver> relocalizer::setup_pnp_solver(const std::vector<unsigned int>& valid_indices,
                                                                 const eigen_alloc_vector<Vec3_t>& bearings,
                                                                 const std::vector<cv::KeyPoint>& keypts,
                                                                 const std::vector<std::shared_ptr<data::landmark>>& matched_landmarks,
                                                                 const std::vector<float>& scale_factors) const {
    // Resample valid elements
    const auto valid_bearings = util::resample_by_indices(bearings, valid_indices);
    const auto valid_keypts = util::resample_by_indices(keypts, valid_indices);
    const auto valid_assoc_lms = util::resample_by_indices(matched_landmarks, valid_indices);
    eigen_alloc_vector<Vec3_t> valid_landmarks(valid_indices.size());
    for (unsigned int i = 0; i < valid_indices.size(); ++i) {
        valid_landmarks.at(i) = valid_assoc_lms.at(i)->get_pos_in_world();
    }
    // Setup PnP solver
    return std::make_unique<solve::pnp_solver>(valid_bearings, valid_keypts, valid_landmarks, scale_factors, 10, use_fixed_seed_);
}
[[maybe_unused]] void relocalizer::visualCandidates(const std::vector<std::shared_ptr<data::keyframe>>& reloc_candidates) {
    cv::Mat show_img;
    bool first_img = true;
    for (const auto& reloc_candidate : reloc_candidates) {
        if (first_img) {
            show_img = reloc_candidate->img.clone();
            first_img = false;
        }
        else {
            cv::hconcat(show_img, reloc_candidate->img.clone(), show_img);
        }
    }
    cv::imwrite("relocalized candidates", show_img);
}
[[maybe_unused]] void relocalizer::drawMatches(const cv::Mat& img1, const cv::Mat& img2,
                              const std::vector<cv::KeyPoint>& undist_keypts_1, const std::vector<cv::KeyPoint>& undist_keypts_2,
                              const std::vector<cv::DMatch>& matches) {
    cv::Mat show_img;
    cv::drawMatches(img1, undist_keypts_1, img2, undist_keypts_2, matches, show_img);
    cv::imwrite("matches_w_candidates.png", show_img);
}
const std::shared_ptr<match::sg_matcher>& relocalizer::getSgMatcher() const {
    return sg_matcher_;
}

} // namespace stella_vslam::module
