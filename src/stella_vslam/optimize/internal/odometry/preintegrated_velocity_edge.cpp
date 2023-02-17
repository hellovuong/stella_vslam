//
// Created by vuong on 1/16/23.
//

#include "preintegrated_velocity_edge.h"

#include <utility>
#include "stella_vslam/module/odometry/preintegration.hpp"

namespace stella_vslam::optimize::internal::odometry {
preintegrate_velocity_edge::preintegrate_velocity_edge(module::odometry::IntegratedOdometryMeasurement::iomSharedPtr iom_ptr)
    : iom_ptr_(std::move(iom_ptr)) {
    assert(iom_ptr_);
    setInformation(iom_ptr_->getCovInv());
}
bool preintegrate_velocity_edge::read(std::istream& is) {
    for (int i = 0; i < information().rows(); ++i) {
        for (int j = i; j < information().cols(); ++j) {
            is >> information()(i, j);
            if (i != j) {
                information()(j, i) = information()(i, j);
            }
        }
    }
    return true;
}
bool preintegrate_velocity_edge::write(std::ostream& os) const {
    for (int i = 0; i < information().rows(); ++i) {
        for (int j = i; j < information().cols(); ++j) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}
void preintegrate_velocity_edge::computeError() {
    data::odometry::PoseBiasState poseBiasState0, poseBiasState1;
    warpVerticesToPoseBiasState(poseBiasState0, poseBiasState1);

    _error = iom_ptr_->computeResidual(poseBiasState0, poseBiasState1);
}
void preintegrate_velocity_edge::linearizeOplus() {
    data::odometry::PoseBiasState poseBiasState0, poseBiasState1;
    warpVerticesToPoseBiasState(poseBiasState0, poseBiasState1);

    module::odometry::IntegratedOdometryMeasurement::MatNN J_pose_1, J_pose_2;
    module::odometry::IntegratedOdometryMeasurement::MatN3 J_b_w, J_b_v;

    iom_ptr_->computeJacobianOplus(poseBiasState0, poseBiasState1, J_pose_1, J_b_w, J_b_v, J_pose_2);

    // Map data to J matrix of in base class
    _jacobianOplus[0] = Eigen::Map<module::odometry::IntegratedOdometryMeasurement::MatNN>
                                    (J_pose_1.data(), J_pose_1.rows(), J_pose_1.cols());
    _jacobianOplus[1] = Eigen::Map<module::odometry::IntegratedOdometryMeasurement::MatN3>
                                    (J_b_w.data(), J_b_w.rows(), J_b_w.cols());
    _jacobianOplus[2] = Eigen::Map<module::odometry::IntegratedOdometryMeasurement::MatN3>
                                    (J_b_v.data(), J_b_v.rows(), J_b_v.cols());
    _jacobianOplus[3] = Eigen::Map<module::odometry::IntegratedOdometryMeasurement::MatNN>
                                    (J_pose_2.data(), J_pose_2.rows(), J_pose_2.cols());
}
data::odometry::PoseBiasState preintegrate_velocity_edge::wrapPoseBiasState(const g2o::VertexSE3Expmap* vertexSe3Expmap,
                                                                            const vertex_angular_vel_bias* vertexAngularVelBias,
                                                                            const vertex_linear_vel_bias* vertexLinearVelBias) {
    data::odometry::PoseBiasState poseBiasState;
    Sophus::SO3d rot = Sophus::SO3d(vertexSe3Expmap->estimate().rotation().normalized());
    poseBiasState.setPose(Sophus::SE3d(rot, vertexSe3Expmap->estimate().translation()));
    if (vertexAngularVelBias) {
        poseBiasState.setBiasW(vertexAngularVelBias->estimate());
    }
    else {
        poseBiasState.setBiasW(Eigen::Vector3d::Zero());
    }
    if (vertexLinearVelBias) {
        poseBiasState.setBiasV(vertexLinearVelBias->estimate());
    }
    else {
        poseBiasState.setBiasV(Eigen::Vector3d::Zero());
    }
    return poseBiasState;
}
void preintegrate_velocity_edge::warpVerticesToPoseBiasState(data::odometry::PoseBiasState& poseBiasState0,
                                                             data::odometry::PoseBiasState& poseBiasState1) {
    // State 0
    const auto* vertex_pose_0 = dynamic_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    const auto* vertex_bias_w_0 = dynamic_cast<const vertex_angular_vel_bias*>(_vertices[1]);
    const auto* vertex_bias_v_0 = dynamic_cast<const vertex_linear_vel_bias*>(_vertices[2]);
    // State 1
    const auto* vertex_pose_1 = dynamic_cast<const g2o::VertexSE3Expmap*>(_vertices[3]);
    // Pose and Bias 0, 1
    poseBiasState0 = wrapPoseBiasState(vertex_pose_0, vertex_bias_w_0, vertex_bias_v_0);
    poseBiasState1 = wrapPoseBiasState(vertex_pose_1);
}
} // namespace stella_vslam::optimize::internal::odometry