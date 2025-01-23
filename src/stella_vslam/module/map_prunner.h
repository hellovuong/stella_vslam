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

    void run();

private:
    //-----------------------------------------
    // Memeber methods for keyframes pruning

    /**
     * @brief Algorithm for managing keyframes (views) in a visual SLAM or mapping system.
     *
     * This algorithm identifies and removes redundant views (keyframes) from the map
     * to optimize storage and computation while maintaining the performance of the system.
     * It evaluates the importance of views based on observation statistics, relocalization
     * usefulness, and spatial distribution.
     *
     * @details
     * The algorithm operates as follows:
     * - If the total number of views is less than or equal to a predefined minimum
     *   (`MIN_VIEWS`), no views are deleted.
     * - A subset of views to keep (`V_keep`) is initialized with views created in the
     *   current run that have been observed at least once.
     * - For each remaining view, a score is computed based on:
     *   - `reloc`: Whether the view was used for relocalization.
     *   - `n_obs_cur / max_obs`: The ratio of the number of observations in the current run
     *     to the maximum number of observations across all views.
     *   - `n_obs_runs / n_runs`: The fraction of runs in which the view was observed.
     *   Each factor is weighted by predefined constants (`W1`, `W2`, `W3`).
     * - Views with scores exceeding a predefined threshold (`SCORE_THRESHOLD`) are added to `V_keep`.
     * - Views not in `V_keep` are considered for deletion. These views are sorted by score
     *   in ascending order.
     * - To ensure a uniform spatial distribution, each view in the deletion set is checked
     *   for the number of spatially nearby neighbors (`numNearestNeighbors`). If the number
     *   of neighbors is below a threshold (`NN_THRESHOLD`), the view is retained.
     * - The final set of views for deletion (`V_delete`) is returned.
     *
     * @param V A set of all views (keyframes) in the map.
     * @param MIN_VIEWS The minimum number of views required to avoid pruning.
     * @param SCORE_THRESHOLD The score threshold to decide which views to keep.
     * @param W1 Weight for the relocalization factor in the score.
     * @param W2 Weight for the current observation factor in the score.
     * @param W3 Weight for the run observation factor in the score.
     * @param NN_THRESHOLD Minimum number of spatially nearby views required for a view to be deleted.
     *
     * @note https://arxiv.org/pdf/1908.03605
     * @return V_delete A set of views to delete based on the above criteria.
     */
    std::unordered_map<unsigned int, double> select_view_to_prune(std::unordered_map<unsigned int, std::shared_ptr<data::keyframe>>& V);

    /**
     * @brief Delete set of given keyframes
     *
     * @param D
     */
    void delete_keyframes(const std::unordered_map<unsigned int, double>& D);

    //-----------------------------------------
    // Memeber methods for landmark pruning
    enum class TriangulationResult {
        SUCCESSFUL,
        UNOBSERVABLE,
        TOO_FEW_MEASUREMENTS
    };

    /**
     * @brief Clean up weak landmarks
     */
    void clean_up_landmarks();

    /// brief Triangulate a 3d point from a set of n keypoint measurements as
    ///       bearing vectors.
    /// @param t_G_bv Back-projected bearing vectors from visual frames to
    ///               observations, expressed in the global frame.
    /// @param p_G_C Global positions of visual frames (cameras).
    /// @param p_G_P Triangulated point in global frame.
    /// @return Was the triangulation successful?
    //-----------------------------------------
    TriangulationResult triangulate_from_n_views(const Eigen::Matrix3Xd& t_G_bv, const Eigen::Matrix3Xd& p_G_C, Eigen::Vector3d* p_G_P);

    /**
     * @brief Determines if a visual landmark is well-constrained based on
     *        its observations and geometric properties.
     *
     * This method evaluates the quality of a given visual landmark in the map
     * by verifying its constraints from multiple observations. A landmark is
     * considered well-constrained if:
     *
     * 1. It is observed by a sufficient number of unique keyframes (backlinks).
     * 2. The reprojection error of its position in the camera frame is within
     *    an acceptable threshold for all observations.
     * 3. The landmark's distance from its closest observer lies within a
     *    defined range.
     * 4. The angular disparity (disparity angle) of incidence rays from
     *    different observing cameras is large enough to ensure triangulation
     *    accuracy.
     *
     * @param[in] map The VIMap object containing the map data, including
     *                vertices, observations, and transformations.
     * @param[in] landmark The landmark to evaluate, containing its position,
     *                     observations, and other properties.
     *
     * @return True if the landmark is well-constrained, false otherwise.
     *
     * ### Constraints Evaluated
     * - **Number of Observations**:
     *   - The landmark must be observed by at least `FLAGS_elq_min_observers`
     *     cameras.
     * - **Reprojection Error**:
     *   - For each observation, the reprojection error (squared) in pixels must
     *     not exceed `FLAGS_elq_max_reprojection_error_px^2`.
     * - **Distance Constraints**:
     *   - The signed distance of the landmark from its closest observing camera
     *     must be within the range `[FLAGS_elq_min_distance_from_closest_observer,
     *     FLAGS_elq_max_distance_from_closest_observer]`.
     * - **Angular Disparity**:
     *   - The maximum angular disparity between normalized incidence rays
     *     from observing cameras must exceed `FLAGS_elq_min_observation_angle_deg`.
     *
     * ### Mathematical Overview
     * 1. For each observation:
     *    - Compute the normalized incidence ray \( \vec{r}_i = \frac{p_G - p_{G_C}}{\|p_G - p_{G_C}\|} \),
     *      where \( p_G \) is the landmark position in the global frame and \( p_{G_C} \)
     *      is the position of the observing camera in the global frame.
     *    - Check the reprojection error of the observed keypoint.
     *    - Update the signed distance to the closest observing camera.
     * 2. After processing all observations:
     *    - Compute the maximum disparity angle between incidence rays.
     *    - Verify that the distance and angle constraints are satisfied.
     *
     */
    bool is_well_constrained_lm(std::shared_ptr<data::landmark>);

    /**
     * @brief Computes the maximum angular disparity (in radians) between a bundle of unit incidence rays.
     *
     * This method calculates the largest angular separation between pairs of unit vectors
     * (incidence rays) in a given bundle. The angular disparity is computed based on the
     * cosine of the angle between the vectors and is returned in radians.
     *
     * @details
     * The algorithm iterates through all unique pairs of unit vectors in the input and
     * calculates the absolute cosine of the angle between each pair:
     * - The cosine of the angle between two vectors, \f$ \mathbf{u} \f$ and \f$ \mathbf{v} \f$,
     *   is given by:
     *   \f[
     *   \cos(\theta) = |\mathbf{u} \cdot \mathbf{v}|
     *   \f]
     * - The minimum cosine value (i.e., the smallest similarity between any pair) is tracked.
     * - Finally, the angle corresponding to this minimum cosine is calculated as:
     *   \f[
     *   \theta = \arccos(\text{min\_cos\_angle})
     *   \f]
     *
     * If the input bundle contains fewer than 2 unit vectors, the function returns 0.0
     * since no meaningful disparity can be computed.
     *
     * @param unit_incidence_rays A vector of unit vectors (\f$ \mathbf{u} \f$), where each
     * vector represents a direction in 3D space.
     *
     * @return The maximum angular disparity (in radians) between the unit vectors in the bundle.
     * If there are fewer than 2 vectors, the return value is 0.0.
     *
     * @note The input vectors should be normalized (i.e., have a magnitude of 1) to ensure
     * correct results. Non-normalized vectors may lead to incorrect angular computations.
     */
    double getMaxDisparityRadAngleOfUnitVectorBundle(const std::vector<Eigen::Vector3d>& unit_incidence_rays);

    // Memeber to get Views from map

    //! map database
    data::map_database* map_db_ = nullptr;

    //! bow database
    data::bow_database* bow_db_ = nullptr;

    //-----------------------------------------
    // Pruning keyframes parameters - Sections IV.A

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

    //-----------------------------------------
    // Pruning landmark parameters
    static constexpr uint32_t min_observers = 4u;
    static constexpr double min_distance_from_closest_observer = 0.05;
    static constexpr double max_distance_from_closest_observer = 10;
    static constexpr double min_observation_angle_deg = 5;
};
} // namespace module
} // namespace stella_vslam
