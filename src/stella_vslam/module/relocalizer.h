#ifndef STELLA_VSLAM_MODULE_RELOCALIZER_H
#define STELLA_VSLAM_MODULE_RELOCALIZER_H

#include "stella_vslam/match/bow_tree.h"
#include "stella_vslam/match/projection.h"
#include "stella_vslam/match/robust.h"
#include "stella_vslam/match/bruce_force.h"
#include "stella_vslam/optimize/pose_optimizer.h"
#include "stella_vslam/solve/pnp_solver.h"

#include <memory>

namespace stella_vslam {

namespace data {
class frame;
class base_place_recognition;
class bow_database;
class hf_net_database;
} // namespace data

namespace module {

class relocalizer {
public:
    //! Constructor
    explicit relocalizer(std::shared_ptr<optimize::pose_optimizer> pose_optimizer,
                         double bow_match_lowe_ratio = 0.75, double proj_match_lowe_ratio = 0.9,
                         double robust_match_lowe_ratio = 0.8,
                         unsigned int min_num_bow_matches = 20, unsigned int min_num_valid_obs = 50,
                         bool use_fixed_seed = false,
                         const YAML::Node& yaml_node = YAML::Node());

    explicit relocalizer(const std::shared_ptr<optimize::pose_optimizer>& pose_optimizer, const YAML::Node& yaml_node);

    //! Destructor
    virtual ~relocalizer();

    //! Relocalize the specified frame
    bool relocalize(data::base_place_recognition* bow_db, data::frame& curr_frm) const;

    //! Relocalize the specified frame by given candidates list
    bool reloc_by_candidates(data::frame& curr_frm,
                             const std::vector<std::shared_ptr<stella_vslam::data::keyframe>>& reloc_candidates,
                             bool use_robust_matcher = false) const;
    bool reloc_by_candidate(data::frame& curr_frm,
                            const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                            bool use_robust_matcher) const;
    bool relocalize_by_pnp_solver(data::frame& curr_frm,
                                  const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                                  bool use_robust_matcher,
                                  std::vector<unsigned int>& inlier_indices,
                                  std::vector<std::shared_ptr<data::landmark>>& matched_landmarks) const;
    bool optimize_pose(data::frame& curr_frm,
                       const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                       std::vector<bool>& outlier_flags) const;
    bool refine_pose(data::frame& curr_frm,
                     const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                     const std::set<std::shared_ptr<data::landmark>>& already_found_landmarks) const;
    bool refine_pose_by_local_map(data::frame& curr_frm,
                                  const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm) const;
    /**
     * For debug only
     * @param reloc_candidates
     */
    [[maybe_unused]] static void visualCandidates(const std::vector<std::shared_ptr<data::keyframe>>& reloc_candidates);

    /**
     * For debug only
     * @param reloc_candidates
     */
    [[maybe_unused]] static void drawMatches(const cv::Mat& img1, const cv::Mat& img2,
                                             const std::vector<cv::KeyPoint>& undist_keypts_1, const std::vector<cv::KeyPoint>& undist_keypts_2,
                                             const std::vector<cv::DMatch>& matches_result);

    /**
     * For vis and debug only
     */
    cv::Mat curr_img_;
    [[nodiscard]] const std::shared_ptr<match::sg_matcher>& getSgMatcher() const;

private:
    //! Extract valid (non-deleted) landmarks from landmark vector
    static std::vector<unsigned int> extract_valid_indices(const std::vector<std::shared_ptr<data::landmark>>& landmarks);

    //! Setup PnP solver with the specified 2D-3D matches
    [[nodiscard]] std::unique_ptr<solve::pnp_solver> setup_pnp_solver(const std::vector<unsigned int>& valid_indices,
                                                        const eigen_alloc_vector<Vec3_t>& bearings,
                                                        const std::vector<cv::KeyPoint>& keypts,
                                                        const std::vector<std::shared_ptr<data::landmark>>& matched_landmarks,
                                                        const std::vector<float>& scale_factors) const;

    //! minimum threshold of the number of BoW matches
    const unsigned int min_num_bow_matches_;
    //! minimum threshold of the number of valid (= inlier after pose optimization) matches
    const unsigned int min_num_valid_obs_;

    //! BoW matcher
    const match::bow_tree bow_matcher_;
    //! projection matcher
    const match::projection proj_matcher_;
    //! robust matcher
    const match::robust robust_matcher_;

    //! super glue matcher
    std::shared_ptr<match::sg_matcher> sg_matcher_ = nullptr;

    //! pose optimizer
    std::shared_ptr<optimize::pose_optimizer> pose_optimizer_ = nullptr;

    //! Use fixed random seed for RANSAC if true
    const bool use_fixed_seed_;
};

} // namespace module
} // namespace stella_vslam

#endif // STELLA_VSLAM_MODULE_RELOCALIZER_H
