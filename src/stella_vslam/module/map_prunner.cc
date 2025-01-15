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
      min_views_(yaml_node["min_views"].as<unsigned int>(100)),
      nn_thrs_(yaml_node["nn_thrs"].as<unsigned int>(5)),
      nn_voxel_size_(yaml_node["nn_voxel_size"].as<std::vector<double>>(std::vector<double>{1, 1, 2})),
      score_thrs_(yaml_node["score_thrs"].as<double>(0.5)),
      w1_(yaml_node["w1"].as<double>(1.5)),
      w2_(yaml_node["w2"].as<double>(1.5)),
      w3_(yaml_node["w3"].as<double>(0.2)) {
    spdlog::info("Construct map_prunner module with parameters min_views {}, nn_thrs {}; score_thrs{}; w1 {}; w2 {}", min_views_, nn_thrs_, score_thrs_, w1_, w2_);
}

std::vector<std::pair<unsigned int, double>> map_prunner::select_view_to_prune(std::unordered_map<unsigned int, std::shared_ptr<data::keyframe>>& V_check) {
    // skip prunning if not enough views
    if (V_check.size() <= min_views_) {
        return {};
    }
    // Candidates to be deleted (id and score)
    std::unordered_map<unsigned int, double> D_candidates;
    // maximum number of observations of any view in the current run
    int max_obs = map_db_->get_max_obs();

    if (max_obs == 0) {
        return {};
    }

    for (auto& [id, v] : V_check) {
        bool prune{false};

        // neither origin kf nor already marked will_be_erased nor root node be deleted
        if (!v) {
            continue;
        }

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
        double score = w1_ * reloc + w2_ * ((double)n_obs_cur / max_obs) + w3_ * (n_obs_runs / n_runs);
        // check score threshold
        if (score > score_thrs_) {
            spdlog::info("{} prunning kf {} with score {} <= {} - was obs {} times in this run (max {}) and {} used for reloc/LC",
                         prune ? "" : "not", id, score, score_thrs_, v->get_n_obs_run_id(map_db_->run_), max_obs, v->is_reloc_by() ? "" : "not");
            continue;
        }
        // count nn view
        auto filtered_v_s = map_db_->get_close_keyframes(v->get_pose_cw(), sqrt(nn_voxel_size_.at(0) * nn_voxel_size_.at(0) + nn_voxel_size_.at(1) * nn_voxel_size_.at(1)),
                                                         nn_voxel_size_.at(2));
        unsigned int num_nn{0};
        // do not count the one that will be deleted
        for (const auto& filtered_v : filtered_v_s) {
            if (!filtered_v->will_be_erased() && !D_candidates.count(filtered_v->id_)) {
                num_nn++;
            }
        }

        // check nn_thrs
        if (num_nn >= nn_thrs_) {
            D_candidates[id] = score;
            prune = true;
        }

        spdlog::info("{} prunning kf {} with score {} <= {} - was obs {} times in this run (max {}) and {} used for reloc/LC and nn {} < {}",
                     prune ? "" : "not", id, score, score_thrs_, v->get_n_obs_run_id(map_db_->run_), v->is_reloc_by() ? "" : "not", num_nn, nn_thrs_);
    }

    // how many do you want to get rid of?
    size_t num_delete = map_db_->get_num_keyframes() - min_views_;

    // only find this much!
    if (num_delete >= D_candidates.size()) {
        return {D_candidates.begin(), D_candidates.end()};
    }

    // sort acsent by score
    std::vector<std::pair<unsigned int, double>> sorted_D_candidates(D_candidates.begin(), D_candidates.end());
    // Sort the vector by value
    std::sort(sorted_D_candidates.begin(), sorted_D_candidates.end(),
              [](const std::pair<unsigned int, unsigned int>& a, const std::pair<unsigned int, unsigned int>& b) {
                  return a.second < b.second; // Ascending order by value
              });
    // resize to contain at most num_delete
    sorted_D_candidates.resize(num_delete);

    return sorted_D_candidates;
}

void map_prunner::run() {
    if (map_db_->run_ == 0)
        return;

    if (map_db_->get_num_keyframes() <= min_views_) {
        return;
    }

    for (int i_run = 0; i_run <= map_db_->run_; i_run++) {
        if (map_db_->get_num_keyframes() <= min_views_) {
            return;
        }
        auto last_run_created_keyfrms = map_db_->get_keyframes_by_run(i_run);
        auto D = select_view_to_prune(last_run_created_keyfrms);
        spdlog::info("Will delete {} out of {} view from run {}", D.size(), last_run_created_keyfrms.size(), i_run);
        for (const auto& [id, score] : D) {
            spdlog::info("Delete view {} and its associate", id);
            auto v = map_db_->get_keyframe(id);
            const auto v_landmarks = v->get_landmarks();
            v->prepare_for_erasing(map_db_, bow_db_);

            for (const auto& lm : v_landmarks) {
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
}
} // namespace module
} // namespace stella_vslam
