//
// Created by vuong on 1/16/23.
//

#ifndef STELLA_VSLAM_OPTIMIZE_INTERNAL_ODOMETRY_VERTEX_BIAS_H
#define STELLA_VSLAM_OPTIMIZE_INTERNAL_ODOMETRY_VERTEX_BIAS_H

#include <Eigen/Core>
#include <g2o/core/base_vertex.h>

namespace stella_vslam::optimize::internal::odometry {
class vertex_angular_vel_bias : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    vertex_angular_vel_bias() = default;
    explicit vertex_angular_vel_bias(const Eigen::Vector3d& estimation);

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void setToOriginImpl() override;

    void oplusImpl(const double* update) override;
};
class vertex_linear_vel_bias : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    vertex_linear_vel_bias() = default;
    explicit vertex_linear_vel_bias(Eigen::Vector3d& estimation);

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void setToOriginImpl() override;

    void oplusImpl(const double* update) override;
};

} // namespace stella_vslam::optimize::internal::odometry

#endif // STELLA_VSLAM_OPTIMIZE_INTERNAL_ODOMETRY_VERTEX_BIAS_H
