//
// Created by vuong on 12/24/22.
//
#include "stella_vslam/data/odometry_type.hpp"
#include "stella_vslam/util/sophus_utils.hpp"

namespace stella_vslam::data::odometry {
void DeltaState::update(const Eigen::Vector3d& corrected_b_v, const Eigen::Vector3d& corrected_b_w, double dt) {
    delta_R = Sophus::SO3d::exp(corrected_b_w * dt);

    delta.translation() += delta.so3() * corrected_b_v * dt;
    delta.so3() *= delta_R;

    Sophus::rightJacobianSO3(corrected_b_w * dt, J_r_delta);
}
} // namespace stella_vslam::data::odometry