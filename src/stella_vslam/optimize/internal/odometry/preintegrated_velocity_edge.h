//
// Created by vuong on 1/16/23.
//

#ifndef STELLA_VSLAM_OPTIMIZED_INTERNAL_ODOMETRY_PREINTEGRATED_VELOCITY_EDGE_H
#define STELLA_VSLAM_OPTIMIZED_INTERNAL_ODOMETRY_PREINTEGRATED_VELOCITY_EDGE_H

#include <g2o/core/base_multi_edge.h>
#include <g2o/types/sba/vertex_se3_expmap.h>

#include "stella_vslam/module/odometry/preintegration.hpp"
#include "vertex_bias.h"

namespace stella_vslam::optimize::internal::odometry {
using VecN = Eigen::Matrix<double, 6, 1>;
/**
 * This edge is for preintegration velocity.
 * Implemented from paper: Wisth, D., Camurri, M., & Fallon, M. (2019).
 * Preintegrated Velocity Bias Estimation to Overcome Contact Nonlinearities in Legged Robot Odometry.
 * ArXiv. https://doi.org/10.1109/ICRA40945.2020.9197214
 */
class preintegrate_velocity_edge final : public g2o::BaseMultiEdge<6, VecN> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    preintegrate_velocity_edge() = default;
    explicit preintegrate_velocity_edge(module::odometry::IntegratedOdometryMeasurement::iomSharedPtr iom_pt);

    bool read(std::istream& is);
    bool write(std::ostream& os) const;

    void computeError();
    void linearizeOplus();

private:
    /**
     * @brief Warp Vertices Pose and Vertices Bias in this edge to PoseBiasState type
     * @param poseBiasState0 [out] Pose Bias State
     * @param poseBiasState1 [out] Pose Bias State
     * @note See function: data::odometry::PoseBiasState wrapPoseBiasState
     */
    void warpVerticesToPoseBiasState(data::odometry::PoseBiasState& poseBiasState0, data::odometry::PoseBiasState& poseBiasState1);

    /**
     * @brief wrap vertex pose and vertex bias to type PoseBiasState for convenience
     * @param vertexSe3Expmap
     * @param vertexAngularVelBias
     * @param vertexLinearVelBias
     * @return wrapped in form of PoseBiasState type
     */
    static data::odometry::PoseBiasState wrapPoseBiasState(const g2o::VertexSE3Expmap* vertexSe3Expmap,
                                                           const vertex_angular_vel_bias* vertexAngularVelBias = nullptr,
                                                           const vertex_linear_vel_bias* vertexLinearVelBias = nullptr);

    module::odometry::IntegratedOdometryMeasurement::iomSharedPtr iom_ptr_ = nullptr;
};
} // namespace stella_vslam

#endif // STELLA_VSLAM_OPTIMIZED_INTERNAL_ODOMETRY_PREINTEGRATED_VELOCITY_EDGE_H
