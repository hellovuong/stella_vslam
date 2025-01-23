#include <spdlog/spdlog.h>
#include <stella_vslam/module/map_prunner.h>
#include <stella_vslam/data/keyframe.h>
#include <stella_vslam/data/map_database.h>
#include <stella_vslam/data/landmark.h>
#include <cmath>
#include <vector>

namespace stella_vslam {
namespace module {
map_prunner::map_prunner(const YAML::Node& yaml_node, data::map_database* map_db, data::bow_database* bow_db)
    : map_db_(map_db),
      bow_db_(bow_db),
      min_views_(yaml_node["min_views"].as<unsigned int>(200)),
      nn_thrs_(yaml_node["nn_thrs"].as<unsigned int>(5)),
      nn_voxel_size_(yaml_node["nn_voxel_size"].as<std::vector<double>>(std::vector<double>{1, 1, 2})),
      score_thrs_(yaml_node["score_thrs"].as<double>(0.6)),
      w1_(yaml_node["w1"].as<double>(1.5)),
      w2_(yaml_node["w2"].as<double>(1.5)),
      w3_(yaml_node["w3"].as<double>(1)) {
    spdlog::info("map_prunner: min_views {}, nn_thrs {}; score_thrs{}; w1 {}; w2 {}", min_views_, nn_thrs_, score_thrs_, w1_, w2_);
}

std::unordered_map<unsigned int, double> map_prunner::select_view_to_prune(std::unordered_map<unsigned int, std::shared_ptr<data::keyframe>>& V_check) {
    // Candidates to be deleted (id and score)
    std::vector<std::pair<unsigned int, double>> D_candidates;
    // maximum number of observations of any view in the current run
    int max_obs_cur = map_db_->get_max_obs();

    if (max_obs_cur == 0) {
        return {};
    }

    for (auto& [id, v] : V_check) {
        if (!v) {
            continue;
        }

        // neither origin kf nor already marked will_be_erased nor root node be deleted
        if (v->id_ == 0 || v->will_be_erased() || v->graph_node_->is_spanning_root()) {
            continue;
        }

        // is this view was used for reloc or loop closure
        int reloc = v->is_reloc_by() ? 1 : 0;
        // number of observations of the view in the current run
        int n_obs_cur = v->get_n_obs_run_id(map_db_->run_);
        // number of runs where the view was observed
        double n_obs_runs = v->get_n_obs_runs();
        // total number of runs when the view was present in the component
        double n_runs = map_db_->run_ - v->get_run() + 1;

        // compute score
        double score = w1_ * reloc + w2_ * ((double)n_obs_cur / max_obs_cur) + w3_ * (n_obs_runs / n_runs);
        // check score threshold
        if (score >= score_thrs_) {
            spdlog::debug("Not prunning kf {} with score {} >= {} - was obs {} times in this run (max {}) and {} used for reloc/LC",
                          id, score, score_thrs_, v->get_n_obs_run_id(map_db_->run_), max_obs_cur, v->is_reloc_by() ? "" : "not");
            continue;
        }
        D_candidates.push_back({id, score});
    }

    // how many do you want to get rid of?
    size_t num_delete = map_db_->get_num_keyframes() - min_views_;

    // sort acsent by score
    // Sort the vector by value
    std::sort(D_candidates.begin(), D_candidates.end(),
              [](const std::pair<unsigned int, double>& a, const std::pair<unsigned int, double>& b) {
                  return a.second < b.second; // Ascending order by value
              });

    // check sorted Candidates with nn threshold
    std::unordered_map<unsigned int, double> D; // list to be deleted
    D.clear();
    for (const auto& [id, score] : D_candidates) {
        bool prune = false;
        auto v = map_db_->get_keyframe(id);
        // count nn view
        auto filtered_v_s = map_db_->get_close_keyframes(v->get_pose_cw(), sqrt(nn_voxel_size_.at(0) * nn_voxel_size_.at(0) + nn_voxel_size_.at(1) * nn_voxel_size_.at(1)),
                                                         nn_voxel_size_.at(2));
        unsigned int num_nn{0};
        // do not count the one that will be deleted
        for (const auto& filtered_v : filtered_v_s) {
            if (!filtered_v->will_be_erased() && !D.count(filtered_v->id_)) {
                num_nn++;
            }
        }

        // check nn_thrs
        if (num_nn >= nn_thrs_) {
            D[id] = score;
            prune = true;
        }

        spdlog::debug("{} prunning kf {} with score {} <= {} - was obs {} times in this run (max {}) and {} used for reloc/LC and nn {} < {}",
                      prune ? "" : "not", id, score, score_thrs_, v->get_n_obs_run_id(map_db_->run_), max_obs_cur, v->is_reloc_by() ? "" : "not", num_nn, nn_thrs_);

        if (D.size() >= num_delete)
            break;
    }

    return D;
}

void map_prunner::run() {
    // save some info for log
    auto num_kfs_before = map_db_->get_num_keyframes();
    auto num_lms_before = map_db_->get_num_landmarks();

    if (map_db_->run_ > 0 && map_db_->get_num_keyframes() >= min_views_) {
        for (int i_run = 0; i_run <= map_db_->run_; i_run++) {
            spdlog::info("Checking views from run {}", i_run);
            auto last_run_created_keyfrms = map_db_->get_keyframes_by_run(i_run);
            auto D = select_view_to_prune(last_run_created_keyfrms);
            spdlog::debug("Deleting {} out of {} views from run {}", D.size(), last_run_created_keyfrms.size(), i_run);
            delete_keyframes(D);
        }
    }

    clean_up_landmarks();

    spdlog::info("Before prune: total kfs {} and lms {}", num_kfs_before, num_lms_before);
    spdlog::info("After prune:  total kfs {} and lms {}", map_db_->get_num_keyframes(), map_db_->get_num_landmarks());
}

void map_prunner::delete_keyframes(const std::unordered_map<unsigned int, double>& D) {
    for (const auto& [id, score] : D) {
        spdlog::debug("Delete view {} and its associate", id);
        auto keyfrm = map_db_->get_keyframe(id);
        const auto landmarks = keyfrm->get_landmarks();
        keyfrm->prepare_for_erasing(map_db_, bow_db_);

        for (const auto& lm : landmarks) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }
            if (!lm->has_representative_descriptor()) {
                lm->compute_descriptor();
            }
            if (!lm->has_valid_prediction_parameters()) {
                lm->update_mean_normal_and_obs_scale_variance();
            }
        }
    }
}

map_prunner::TriangulationResult map_prunner::triangulate_from_n_views(const Eigen::Matrix3Xd& t_G_bv, const Eigen::Matrix3Xd& p_G_C, Eigen::Vector3d* p_G_P) {
    const int num_measurements = t_G_bv.cols();
    if (num_measurements < 2) {
        return TriangulationResult::TOO_FEW_MEASUREMENTS;
    }

    // 1.) Formulate the geometrical problem
    // p_G_P + alpha[i] * t_G_bv[i] = p_G_C[i]      (+ alpha intended)
    // as linear system Ax = b, where
    // x = [p_G_P; alpha[0]; alpha[1]; ... ] and b = [p_G_C[0]; p_G_C[1]; ...]
    //
    // 2.) Apply the approximation AtAx = Atb
    // AtA happens to be composed of mostly more convenient blocks than A:
    // - Top left = N * Eigen::Matrix3d::Identity()
    // - Top right and bottom left = t_G_bv
    // - Bottom right = t_G_bv.colwise().squaredNorm().asDiagonal()

    // - Atb.head(3) = p_G_C.rowwise().sum()
    // - Atb.tail(N) = columnwise dot products between t_G_bv and p_G_C
    //               = t_G_bv.cwiseProduct(p_G_C).colwise().sum().transpose()
    //
    // 3.) Apply the Schur complement to solve after p_G_P only
    // AtA = [E B; C D] (same blocks as above) ->
    // (E - B * D.inverse() * C) * p_G_P = Atb.head(3) - B * D.inverse() * Atb.tail(N)

    const Eigen::MatrixXd BiD = t_G_bv * t_G_bv.colwise().squaredNorm().asDiagonal().inverse();
    const Eigen::Matrix3d AxtAx = num_measurements * Eigen::Matrix3d::Identity() - BiD * t_G_bv.transpose();
    const Eigen::Vector3d Axtbx = p_G_C.rowwise().sum() - BiD * t_G_bv.cwiseProduct(p_G_C).colwise().sum().transpose();

    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr = AxtAx.colPivHouseholderQr();
    static constexpr double kRankLossTolerance = 1e-5;
    qr.setThreshold(kRankLossTolerance);
    const size_t rank = qr.rank();
    if (rank < 3) {
        return TriangulationResult::UNOBSERVABLE;
    }

    *p_G_P = qr.solve(Axtbx);
    return TriangulationResult::SUCCESSFUL;
}

bool map_prunner::is_well_constrained_lm(std::shared_ptr<data::landmark> landmark) {
    const auto& backlinks = landmark->get_observations();
    if (backlinks.size() < min_observers) {
        return false;
    }

    const Eigen::Vector3d& p_G_fi = landmark->get_pos_in_world();
    std::vector<Eigen::Vector3d> G_normalized_incidence_rays;
    G_normalized_incidence_rays.reserve(backlinks.size());
    double signed_distance_from_closest_observer = std::numeric_limits<double>::max();
    for (const auto& [ptr, id] : backlinks) {
        const auto& vertex = ptr.lock();

        // p_C = R_cw * p_w + t_cw
        // p_W = R_wc * pc + t_wc
        const Eigen::Vector3d p_C_fi = vertex->get_rot_cw() * landmark->get_pos_in_world() + vertex->get_trans_cw();
        const Eigen::Vector3d G_incidence_ray = vertex->get_trans_wc() - p_G_fi;

        const double distance = G_incidence_ray.norm();
        const double signed_distance = distance * (p_C_fi(2) < 0.0 ? -1.0 : 1.0);
        signed_distance_from_closest_observer = std::min(signed_distance_from_closest_observer, signed_distance);

        if (distance > 0) {
            G_normalized_incidence_rays.emplace_back(G_incidence_ray / distance);
        }
        auto undist_keypt = vertex->frm_obs_.undist_keypts_.at(id);
        const auto scale_level = static_cast<unsigned int>(undist_keypt.octave);
        // Reproject and compute visibility
        Vec2_t reproj;
        float x_right;
        const bool in_image = vertex->camera_->reproject_to_image(vertex->get_rot_cw(), vertex->get_trans_cw(), landmark->get_pos_in_world(), reproj, x_right);
        if (!in_image) {
            return false;
        }
        Vec2_t kpt{undist_keypt.pt.x, undist_keypt.pt.y};
        auto reprojection_error_px_sq = (reproj - kpt).squaredNorm();

        constexpr float chi_sq_3D = 7.81473;
        if (chi_sq_3D < reprojection_error_px_sq * vertex->orb_params_->inv_level_sigma_sq_.at(scale_level)) {
            return false;
        }
    }

    if (signed_distance_from_closest_observer > max_distance_from_closest_observer || signed_distance_from_closest_observer < min_distance_from_closest_observer) {
        return false;
    }

    const double max_disparity_angle_rad = getMaxDisparityRadAngleOfUnitVectorBundle(G_normalized_incidence_rays);

    constexpr double kRadToDeg = 180.0 / M_PI;
    double angle_deg = max_disparity_angle_rad * kRadToDeg;
    if (angle_deg < min_observation_angle_deg) {
        return false;
    }

    return true;
}

void map_prunner::clean_up_landmarks() {
    TriangulationResult ret;
    const auto& all_landmarks = map_db_->get_all_landmarks();

    for (auto lm : all_landmarks) {
        if (!lm) {
            continue;
        }

        if (lm->will_be_erased()) {
            continue;
        }
        // The following have one entry per measurement:
        Eigen::Matrix3Xd G_bearing_vectors;
        Eigen::Matrix3Xd p_G_C_vectors;
        Eigen::Matrix3Xd p_G_fi_vectors;

        // [kf, id]
        auto observations = lm->get_observations();

        if (observations.size() < 2) {
            lm->prepare_for_erasing(map_db_);
            continue;
        }

        G_bearing_vectors.conservativeResize(Eigen::NoChange, observations.size());
        p_G_C_vectors.conservativeResize(Eigen::NoChange, observations.size());

        int num_measurements = 0;
        for (auto& [kf, id] : observations) {
            auto C_bearing_vector = kf.lock()->frm_obs_.bearings_.at(id);
            auto G_bearing = kf.lock()->get_rot_cw().transpose() * C_bearing_vector;
            G_bearing_vectors.col(num_measurements) = G_bearing;
            p_G_C_vectors.col(num_measurements) = kf.lock()->get_trans_wc();
            num_measurements++;
        }
        // Resize to final number of valid measurements
        G_bearing_vectors.conservativeResize(Eigen::NoChange, num_measurements);
        p_G_C_vectors.conservativeResize(Eigen::NoChange, num_measurements);

        // return result
        Eigen::Vector3d p_G_fi;
        ret = triangulate_from_n_views(G_bearing_vectors, p_G_C_vectors, &p_G_fi);

        if (ret == TriangulationResult::SUCCESSFUL) {
            lm->set_pos_in_world(p_G_fi);
            lm->update_mean_normal_and_obs_scale_variance();
            if (!is_well_constrained_lm(lm)) {
                lm->prepare_for_erasing(map_db_);
            }
        }
        else {
            lm->prepare_for_erasing(map_db_);
        }
    }
}

double map_prunner::getMaxDisparityRadAngleOfUnitVectorBundle(const std::vector<Eigen::Vector3d>& unit_incidence_rays) {
    if (unit_incidence_rays.size() < 2u) {
        return 0.0;
    }

    double min_cos_angle = 1.0;
    for (size_t i = 0; i < unit_incidence_rays.size(); ++i) {
        for (size_t j = i + 1; j < unit_incidence_rays.size(); ++j) {
            assert(std::abs(unit_incidence_rays[i].squaredNorm() - 1.0) <= 1e-6 && std::abs(unit_incidence_rays[i].squaredNorm() - 1.0) <= 1e-6);
            min_cos_angle = std::min(
                min_cos_angle,
                std::abs(unit_incidence_rays[i].dot(unit_incidence_rays[j])));
        }
    }
    return std::acos(min_cos_angle);
}
} // namespace module
} // namespace stella_vslam
