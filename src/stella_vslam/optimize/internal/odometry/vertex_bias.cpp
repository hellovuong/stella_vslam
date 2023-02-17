//
// Created by vuong on 1/16/23.
//

#include "vertex_bias.h"

namespace stella_vslam::optimize::internal::odometry {

vertex_angular_vel_bias::vertex_angular_vel_bias(const Eigen::Vector3d& estimation) {
    setEstimate(estimation);
}
bool vertex_angular_vel_bias::read(std::istream& is) {
    for (unsigned int i = 0; i < 3; ++i) {
        is >> _estimate(i);
    }
    return true;
}
bool vertex_angular_vel_bias::write(std::ostream& os) const {
    const Eigen::Vector3d bias = estimate();
    for (unsigned int i = 0; i < 3; ++i) {
        os << bias(i) << " ";
    }
    return os.good();
}
void vertex_angular_vel_bias::setToOriginImpl() {
    _estimate.fill(0);
}
void vertex_angular_vel_bias::oplusImpl(const double* update) {
    Eigen::Map<const Eigen::Vector3d> v(update);
    _estimate += v;
}

vertex_linear_vel_bias::vertex_linear_vel_bias(Eigen::Vector3d& estimation) {
    setEstimate(estimation);
}
bool vertex_linear_vel_bias::read(std::istream& is) {
    for (unsigned int i = 0; i < 3; ++i) {
        is >> _estimate(i);
    }
    return true;
}
bool vertex_linear_vel_bias::write(std::ostream& os) const {
    const Eigen::Vector3d bias = estimate();
    for (unsigned int i = 0; i < 3; ++i) {
        os << bias(i) << " ";
    }
    return os.good();
}
void vertex_linear_vel_bias::setToOriginImpl() {
    _estimate.fill(0);
}
void vertex_linear_vel_bias::oplusImpl(const double* update) {
    Eigen::Map<const Eigen::Vector3d> v(update);
    _estimate += v;
}
} // namespace stella_vslam::optimize::internal::odometry