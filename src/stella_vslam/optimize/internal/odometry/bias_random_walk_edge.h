//
// Created by vuong on 1/16/23.
//

#ifndef STELLA_VSLAM_OPTIMIZE_INTERNAL_ODOMETRY_BIAS_RANDOM_WALK_EDGE_H
#define STELLA_VSLAM_OPTIMIZE_INTERNAL_ODOMETRY_BIAS_RANDOM_WALK_EDGE_H

#include <Eigen/Core>
#include <g2o/core/base_binary_edge.h>
#include "vertex_bias.h"

namespace stella_vslam::optimize::internal::odometry {
class linear_vel_bias_random_walk_edge final : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, vertex_linear_vel_bias, vertex_linear_vel_bias> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    linear_vel_bias_random_walk_edge() = default;

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override;

    void linearizeOplus() override;
};

class angular_vel_bias_random_walk_edge final : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, vertex_angular_vel_bias, vertex_angular_vel_bias> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    angular_vel_bias_random_walk_edge() = default;

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override;

    void linearizeOplus() override;
};
} // namespace stella_vslam::optimize::internal::odometry
#endif // STELLA_VSLAM_OPTIMIZE_INTERNAL_ODOMETRY_BIAS_RANDOM_WALK_EDGE_H
