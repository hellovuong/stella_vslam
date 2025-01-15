#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <yaml-cpp/yaml.h>

/**
 * @brief Implementation of View management for lifelong visual maps
 * @note https://arxiv.org/pdf/1908.03605
 */
namespace stella_vslam {
namespace data {
class keyframe;
class landmark;
class map_database;
class bow_database;
} // namespace data

namespace module {
class map_prunner {
public:
    map_prunner() = default;
    map_prunner(const YAML::Node& yaml_node, data::map_database* map_db, data::bow_database* bow_db);
    map_prunner(map_prunner&&) = default;
    map_prunner(const map_prunner&) = default;
    map_prunner& operator=(map_prunner&&) = default;
    map_prunner& operator=(const map_prunner&) = default;
    ~map_prunner() = default;

    std::vector<std::pair<unsigned int, double>> select_view_to_prune(std::unordered_map<unsigned int, std::shared_ptr<data::keyframe>>& V);

    void run();

    bool once_{false};

private:
    //-----------------------------------------
    // Memeber to get Views from map

    //! map database
    data::map_database* map_db_ = nullptr;

    //! bow database
    data::bow_database* bow_db_ = nullptr;

    //-----------------------------------------
    // Pruning Parameters - Sections IV.A

    //! minimum number of views required before the pruning alg. to run
    unsigned int min_views_;

    //! Nearest neighbor threshold; if the total number of views nearby is less than this thrs, the view will NOT marked for deletion
    uint32_t nn_thrs_;

    //! Vector3d (x, y, theta) used for constraining for seach space while finding neighbor view
    std::vector<double> nn_voxel_size_;

    //! Score threshod - if score is less than this, the view will be consider to be deleted
    double score_thrs_;

    //! Weight views that are used for reloc into previous map
    double w1_;

    //! Weight total observation in current run
    double w2_;

    //! Weight total runs where the view was obeserved at least once
    double w3_;
};
} // namespace module
} // namespace stella_vslam
