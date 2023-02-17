//
// Created by vuong on 12/23/22.
//
#ifndef STELLA_VSLAM_DATA_ODOMETRY_TYPE_HPP
#define STELLA_VSLAM_DATA_ODOMETRY_TYPE_HPP

#include <Eigen/Dense>
#include <utility>
#include "se3.hpp"
#include <Eigen/Core>

namespace stella_vslam::data::odometry {
constexpr size_t se3_SIZE = 6; ///< Dimentionality of the pose state
constexpr size_t sim3_SIZE = 7;
constexpr size_t se3_BIAS_SIZE = 12; ///< Dimentionality of the pose-bias state

class PoseBiasState {
public:
    PoseBiasState()
        : PoseBiasState(Sophus::SE3d(), Eigen::Vector3d(), Eigen::Vector3d()){};

    PoseBiasState(const Sophus::SE3d& pose, Eigen::Vector3d bv, Eigen::Vector3d bw)
        : pose(pose),
          bias_v(std::move(bv)),
          bias_w(std::move(bw)){};

    [[nodiscard]] const Sophus::SE3d& getPose() const {
        return pose;
    }
    void setPose(const Sophus::SE3d& pose_) {
        PoseBiasState::pose = pose_;
    }
    [[nodiscard]] const Eigen::Vector3d& getBiasV() const {
        return bias_v;
    }
    void setBiasV(const Eigen::Vector3d& biasV) {
        bias_v = biasV;
    }
    [[nodiscard]] const Eigen::Vector3d& getBiasW() const {
        return bias_w;
    }
    void setBiasW(const Eigen::Vector3d& biasW) {
        bias_w = biasW;
    }

private:
    Sophus::SE3d pose;
    Eigen::Vector3d bias_v;
    Eigen::Vector3d bias_w;
};

/**
 * @brief Delta in SE(3) and right jacobian.
 */
class DeltaState {
public:
    using Ptr = std::shared_ptr<DeltaState>;
    using VecN = Eigen::Matrix<double, se3_BIAS_SIZE, 1>;

    /**
     * @brief Default constructor with Identity pose and zero other values.
     */
    DeltaState()
        : delta(Sophus::SE3d()),
          delta_R(Sophus::SO3d()),
          J_r_delta(Sophus::Matrix3d::Zero()){};

    /// @brief Integrate forward in time given linear/angular velocity in body frame
    /// @param [in] corrected_b_v Corrected linear velocity
    /// @param [in] corrected_b_w Corrected angular velocity
    /// @param [in] dt Delta between measurements in second
    /// @returns update new pose, delta rotation between two newest measurements and right jacobian of delta rotation
    /// @note linear/angular velocity need to corrected (minus bias) before pass to function
    void update(const Eigen::Vector3d& corrected_b_v, const Eigen::Vector3d& corrected_b_w, double dt);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Sophus::SE3d delta;
    Sophus::SO3d delta_R; //! Current delta_R. only use preintegration
    Sophus::Matrix3d J_r_delta;
};

/// @brief Timestamped pose, linear/angular velocity measurements.
struct OdometryData {
    using Ptr = std::shared_ptr<OdometryData>;

    /// @brief Default constructor with zero measurements and identity pose.
    OdometryData()
        : t_s(0),
          linear_velocity(Eigen::Vector3d::Zero()),
          angular_velocity(Eigen::Vector3d::Zero()),
          pose(Sophus::SE3d()) {}

    /// @brief Constructor
    OdometryData(const double ts, const Sophus::SE3d& pose,
                 Eigen::Vector3d linear_vel,
                 Eigen::Vector3d angular_vel)
        : t_s(ts),
          linear_velocity(std::move(linear_vel)),
          angular_velocity(std::move(angular_vel)),
          pose(pose) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double t_s;                       ///< timestamp in seconds
    Eigen::Vector3d linear_velocity;  ///< Linear velocity measurement
    Eigen::Vector3d angular_velocity; ///< Angular velocity measurement
    Sophus::SE3d pose;                ///< Pose measurement
};

} // namespace stella_vslam::data::odometry

#endif // STELLA_VSLAM_DATA_ODOMETRY_TYPE_HPP
