//
// Created by vuong on 12/23/22.
//

#include "preintegration.hpp"
#include "stella_vslam/util/yaml.h"

namespace stella_vslam::module::odometry {
IntegratedOdometryMeasurement::IntegratedOdometryMeasurement(const YAML::Node& yaml_node)
    : IntegratedOdometryMeasurement(util::get_se3<double>(yaml_node["Tbc_"]),
                                    Eigen::Vector3d(),
                                    Eigen::Vector3d(),
                                    util::yaml_optional_ref(yaml_node, "noiseLinearVel").as<double>(0.0f),
                                    util::yaml_optional_ref(yaml_node, "noiseAngularVel").as<double>(0.0f),
                                    util::yaml_optional_ref(yaml_node, "linearVelWalk").as<double>(0.0f),
                                    util::yaml_optional_ref(yaml_node, "angularVelWalk").as<double>(0.0f),
                                    util::yaml_optional_ref(yaml_node, "frequency").as<double>(0.0f)) {}

void IntegratedOdometryMeasurement::integrateMeasurements(const std::vector<data::odometry::OdometryData>& measurements) {
    assert(measurements.size() >= 2);
    for (size_t i = 0; i < measurements.size() - 1; ++i) {
        auto dt = measurements.at(i + 1).t_s - measurements.at(i).t_s;
        integrateMeasurement(measurements.at(i), dt);
        measurements_.push_back(measurements.at(i));
    }
}
void IntegratedOdometryMeasurement::integrateMeasurements(std::deque<data::odometry::OdometryData>& measurements) {
    assert(measurements.size() >= 2);

    // i from 0 -> size - 2, skip last measurement, only take its timestamps
    while (not measurements.empty()) {
        auto measurement_i = measurements.front();
        measurements.pop_front();
        if (measurements.empty()) {
            break;
        }
        auto measurement_i_1 = measurements.front();
        measurements.pop_front();
        auto dt = measurement_i_1.t_s - measurement_i.t_s;
        integrateMeasurement(measurement_i, dt);
        measurements_.push_back(measurement_i);
    }
}
void IntegratedOdometryMeasurement::integrateMeasurement(const data::odometry::OdometryData& measurement,
                                                         const double dt) {
    if (dt <= 0) {
        throw std::runtime_error("IntegratedOdometryMeasurement::integrateMeasurement: dt <= 0");
    }
    // Update preintegrated measurements, Jacobian and covariance
    propagateState(measurement, dt);
}
void IntegratedOdometryMeasurement::propagateState(const data::odometry::OdometryData& measurement,
                                                   const double dt) {
    //! Save current rotation for updating Jacobians
    auto delta_R_ik = delta_state_.delta.so3().matrix();
    //! Correct measurement by bias
    auto correctedLinearVel = correctLinearVel(measurement.linear_velocity);
    auto correctedAngularVel = correctAngularVel(measurement.linear_velocity);
    //! Do update
    delta_t_ += dt;
    delta_state_.update(correctedLinearVel, correctedAngularVel, dt);

    //! Update Covariance
    //! equation [23] & [24]
    updateCovariance(delta_R_ik, correctedLinearVel, dt);
    //! Update Jacobian wrt bias similar to imu preintegration (C. Froster et al).
    updateJacobians(delta_R_ik, correctedLinearVel, dt);
}
void IntegratedOdometryMeasurement::updateCovariance(const Sophus::SO3d::Transformation& delta_R_ik,
                                                     const Eigen::Vector3d& corrected_linear_vel,
                                                     double dt) {
    //! equation 24
    Sophus::Matrix6d A;
    A.setIdentity();
    A.block<3, 3>(0, 0) = delta_state_.delta_R.matrix().transpose();
    A.block<3, 3>(3, 0) = -delta_R_ik * Sophus::SO3d::hat(corrected_linear_vel) * dt;

    //! equation 24
    Sophus::Matrix6d B;
    B.setZero();
    B.block<3, 3>(0, 0) = delta_state_.J_r_delta * dt;
    B.block<3, 3>(3, 3) = delta_R_ik * dt;

    //! equation 23
    cov_ = A * cov_ * A.transpose() + B * sigma_noise_ * B.transpose();

    //! covariance for random walk model
    cov_random_walk_ += sigma_noise_random_walk_;
}
Eigen::Vector3d IntegratedOdometryMeasurement::correctLinearVel(const Eigen::Vector3d& measurement) const {
    return measurement - bias_v_;
}
Eigen::Vector3d IntegratedOdometryMeasurement::correctAngularVel(const Eigen::Vector3d& measurement) const {
    return measurement - bias_w_;
}
void IntegratedOdometryMeasurement::updateJacobians(const Sophus::SO3d::Transformation& delta_R_ik,
                                                    const Eigen::Vector3d& corrected_linear_vel,
                                                    double dt) {
    //! Jacobian translation wrt bias v and w
    d_p_d_bv_ -= delta_R_ik * dt;
    d_p_d_bw_ -= delta_R_ik * Sophus::SO3d::hat(corrected_linear_vel) * d_R_d_bw_;

    //! Jacobian rotation wrt bias v and w
    d_R_d_bw_ = delta_state_.delta_R.matrix().transpose() * delta_state_.J_r_delta * delta_t_ - delta_state_.J_r_delta * dt;
}
void IntegratedOdometryMeasurement::predictState(const data::odometry::PoseBiasState& state0,
                                                 data::odometry::PoseBiasState& state1) {
    std::lock_guard<std::mutex> lock(mutex_iom);
    auto pose0 = state0.getPose();
    const auto& bw0 = state0.getBiasW();
    const auto& bv0 = state0.getBiasW();
    auto Rwb1 = pose0.so3() * getDeltaRotation(bw0);
    auto pwb1 = pose0.translation() + pose0.so3() * getDeltaTranslation(bv0, bw0);
    state1.setPose(Sophus::SE3d(Rwb1, pwb1));
}
Sophus::SO3d IntegratedOdometryMeasurement::getDeltaRotation(const Eigen::Vector3d& bias_w) const {
    auto delta_bw = bias_w - bias_w_;
    return delta_state_.delta.so3() * Sophus::SO3d::exp(d_R_d_bw_ * delta_bw);
}
Sophus::SE3d::TranslationMember IntegratedOdometryMeasurement::getDeltaTranslation(const Eigen::Vector3d& bias_v,
                                                                                   const Eigen::Vector3d& bias_w) const {
    auto delta_bv = bias_v - bias_v_;
    auto delta_bw = bias_w - bias_w_;
    return delta_state_.delta.translation() + d_p_d_bw_ * delta_bw + d_p_d_bv_ * delta_bv;
}
[[maybe_unused]] const Eigen::Matrix3d& IntegratedOdometryMeasurement::getD_R_d_bw() {
    std::lock_guard<std::mutex> lock(mutex_iom);
    return d_R_d_bw_;
}
[[maybe_unused]] const Eigen::Matrix3d& IntegratedOdometryMeasurement::getD_p_d_bw() {
    std::lock_guard<std::mutex> lock(mutex_iom);
    return d_p_d_bw_;
}
[[maybe_unused]] const Eigen::Matrix3d& IntegratedOdometryMeasurement::getD_p_d_bv() {
    std::lock_guard<std::mutex> lock(mutex_iom);
    return d_p_d_bv_;
}
void IntegratedOdometryMeasurement::setUpdatedBias(Eigen::Vector3d& updated_bias_v, Eigen::Vector3d& updated_bias_w) {
    std::lock_guard<std::mutex> lock(mutex_iom);
    setUpdatedBiasV(updated_bias_v);
    setUpdatedBiasW(updated_bias_w);
}
void IntegratedOdometryMeasurement::setUpdatedBiasV(Eigen::Vector3d& updated_bias_v) {
    std::lock_guard<std::mutex> lock(mutex_iom);
    updated_bias_v_ = updated_bias_v;
}
void IntegratedOdometryMeasurement::setUpdatedBiasW(Eigen::Vector3d& updated_bias_w) {
    std::lock_guard<std::mutex> lock(mutex_iom);
    updated_bias_w_ = updated_bias_w;
}
const Sophus::SE3d& IntegratedOdometryMeasurement::getTbc() const {
    return Tbc_;
}
void IntegratedOdometryMeasurement::setSigmaNoiseCov(const double noise_v, const double noise_w,
                                                     const double random_walk_v, const double random_walk_w,
                                                     const double freq) {
    const auto square_root_freq = std::sqrt(freq);

    const auto sigma_noise_v = std::pow(noise_v * square_root_freq, 2);
    const auto sigma_noise_w = std::pow(noise_w * square_root_freq, 2);
    sigma_noise_.diagonal() << Eigen::Vector3d::Constant(sigma_noise_w), Eigen::Vector3d::Constant(sigma_noise_v);

    const auto sigma_random_walk_v = std::pow(random_walk_v / square_root_freq, 2);
    const auto sigma_random_walk_w = std::pow(random_walk_w / square_root_freq, 2);
    sigma_noise_random_walk_.diagonal() << Eigen::Vector3d::Constant(sigma_random_walk_w), Eigen::Vector3d::Constant(sigma_random_walk_v);
}
const Eigen::DiagonalMatrix<double, 6>& IntegratedOdometryMeasurement::getSigmaNoise() const {
    return sigma_noise_;
}
void IntegratedOdometryMeasurement::setSigmaNoise(const Eigen::DiagonalMatrix<double, 6>& sigmaNoise) {
    sigma_noise_ = sigmaNoise;
}
const Eigen::DiagonalMatrix<double, 6>& IntegratedOdometryMeasurement::getSigmaNoiseRandomWalk() const {
    return sigma_noise_random_walk_;
}
void IntegratedOdometryMeasurement::setSigmaNoiseRandomWalk(const Eigen::DiagonalMatrix<double, 6>& sigmaNoiseRandomWalk) {
    sigma_noise_random_walk_ = sigmaNoiseRandomWalk;
}
void IntegratedOdometryMeasurement::mergePrevious(const std::shared_ptr<IntegratedOdometryMeasurement>& prev_iom_ptr) {
    // saved prev/curr measurements for reintegration
    auto prev_measurements = prev_iom_ptr->getMeasurements();
    auto curr_measurements = measurements_;

    // saved updated_bias for reintegration
    auto u_bias_v = getUpdatedBiasV();
    auto u_bias_w = getUpdatedBiasW();

    // reset preintegration, variables = 0, bias = updated_bias
    reset(u_bias_v, u_bias_w);

    // reintegration from prev -> curr
    integrateMeasurements(prev_measurements);
    integrateMeasurements(curr_measurements);
}
const std::vector<data::odometry::OdometryData>& IntegratedOdometryMeasurement::getMeasurements() const {
    return measurements_;
}
const Sophus::SE3d& IntegratedOdometryMeasurement::getTcb() const {
    return Tcb_;
}
IntegratedOdometryMeasurement::VecN IntegratedOdometryMeasurement::computeResidual(const data::odometry::PoseBiasState& state0, const data::odometry::PoseBiasState& state1) const {
    const auto& delta_R = getDeltaRotation(state0.getBiasW());
    const auto& delta_p = getDeltaTranslation(state0.getBiasV(), state0.getBiasW());

    const auto& pose0 = state0.getPose();
    const auto& pose1 = state1.getPose();

    const auto e_r = computeResidualRotation(pose0.so3(), pose1.so3(), delta_R);
    const auto e_p = pose0.so3().inverse() * (pose1.translation() - pose0.translation()) - delta_p;

    VecN residual;
    residual << e_r, e_p;
    return residual;
}
IntegratedOdometryMeasurement::Vec3 IntegratedOdometryMeasurement::computeResidualRotation(const Sophus::SO3d& R0, const Sophus::SO3d& R1, const Sophus::SO3d& dR) {
    return computeResidualRotationSO3(R0, R1, dR).log();
}
Sophus::SO3d IntegratedOdometryMeasurement::computeResidualRotationSO3(const Sophus::SO3d& R0, const Sophus::SO3d& R1, const Sophus::SO3d& dR) {
    return dR.inverse() * R0.inverse() * R1;
}
void IntegratedOdometryMeasurement::computeJacobianOplus(const data::odometry::PoseBiasState& state0, const data::odometry::PoseBiasState& state1,
                                                         MatNN& J_pose_1, MatN3& J_b_w, MatN3& J_b_v, MatNN& J_pose_2) {
    // beforehand computing
    const auto& pose0 = state0.getPose();
    const auto& pose1 = state1.getPose();

    const auto& delta_R = getDeltaRotation(state0.getBiasW());
    const auto& delta_p = getDeltaTranslation(state0.getBiasV(), state0.getBiasW());

    const auto& residual_R = computeResidualRotationSO3(pose0.so3(), pose1.so3(), delta_R);
    const auto& residual_r = residual_R.log();

    const auto& delta_bias_w = state0.getBiasW() - bias_w_;

    Eigen::Matrix3d inv_right_jacobian_residual_r;
    Sophus::rightJacobianInvSO3(residual_r, inv_right_jacobian_residual_r);

    Eigen::Matrix3d right_jacobian_bw;
    Sophus::rightJacobianSO3(d_R_d_bw_ * delta_bias_w, right_jacobian_bw);

    // J[0] - Jacobian wrt pose 1
    J_pose_1.setZero();
    // rotation
    J_pose_1.block<3, 3>(0, 0) = -inv_right_jacobian_residual_r * pose1.so3().matrix().transpose() * pose0.so3().matrix();
    J_pose_1.block<3, 3>(3, 0) = Sophus::SO3d::hat(pose0.so3().inverse() * (pose1.translation() - pose0.translation()));
    // translation
    J_pose_1.block<3, 3>(3, 3) = -pose0.so3().inverse().matrix();

    // J[1] - Jacobian wrt bias angular
    J_b_w.setZero();
    // rotation residual part
    J_b_w.block<3, 3>(0, 0) = -inv_right_jacobian_residual_r * residual_R.matrix().transpose() * right_jacobian_bw * d_R_d_bw_;
    // translation residual part
    J_b_w.block<3, 3>(3, 0) = d_p_d_bw_;

    // J[2] - Jacobian wrt bias linear
    J_b_v.setZero();
    // translation residual part
    J_b_v.block<3, 3>(3, 0) = d_p_d_bv_;

    // J[3] - Jacobian wrt pose 2
    J_pose_2.setZero();
    // rotation
    J_pose_2.block<3, 3>(0, 0) = inv_right_jacobian_residual_r;
    // translation
    J_pose_2.block<3, 3>(3, 0) = pose0.so3().matrix().transpose();
}
} // namespace stella_vslam::module::odometry
