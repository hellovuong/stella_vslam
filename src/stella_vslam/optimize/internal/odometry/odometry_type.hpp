//
// Created by vuong on 12/23/22.
//

#ifndef STELLA_VSLAM_ODOMETRY_TYPE_HPP
#define STELLA_VSLAM_ODOMETRY_TYPE_HPP

#include <Eigen/Dense>
#include <utility>

namespace stella_vslam {
constexpr size_t se3_SIZE = 6; ///< Dimentionality of the pose state
constexpr size_t sim3_SIZE = 7;
constexpr size_t se3_BIAS_SIZE = 12;     ///< Dimentionality of the pose-bias state

/// @brief State that consists of SE(3) pose at a certain time.
struct PoseState {
    using VecN = Eigen::Matrix<double, se3_SIZE, 1>;

    /// @brief Default constructor with Identity pose and zero timestamp.
    PoseState() { t_ns = 0; }

    /// @brief Constructor with timestamp and pose.
    ///
    /// @param t_ns timestamp of the state in nanoseconds
    /// @param T_w_i transformation from the body frame to the world frame
    PoseState(int64_t t_ns, const Sophus::SE3d& T_w_i)
        : t_ns(t_ns), T_w_i(T_w_i) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int64_t t_ns;       ///< timestamp of the state in nanoseconds
    Sophus::SE3d T_w_i; ///< pose of the state
};

/**
 * @brief State that consists of SE(3) pose, linear/angular velocity biases at a certain time.
 */
struct PoseBiasState : public PoseState {
    using Ptr = std::shared_ptr<PoseBiasState>;
    using VecN = Eigen::Matrix<double, se3_BIAS_SIZE, 1>;

    /**
     * @brief Default constructor with Identity pose and zero other values.
     */
    PoseBiasState() {
        bias_v.setZero();
        bias_w.setZero();
    };

    /**
     * @brief Constructor with timestamp, pose, linear velocity, gyroscope and
     * accelerometer biases.
     * @param t_ns timestamp of the state in nanoseconds
     * @param T_w_i transformation from the body frame to the world frame
     * @param vel_w_i linear velocity in world coordinate frame
     * @param bias_linear linear velocity bias
     * @param bias_omega angular velocity bias
     */
    PoseBiasState(int64_t t_ns, const Sophus::SE3d& T_w_i,
                  Eigen::Vector3d bias_v,
                  Eigen::Vector3d bias_w)
        : PoseState(t_ns, T_w_i),
          bias_v(std::move(bias_v)),
          bias_w(std::move(bias_w)) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d bias_v; ///< linear velocity bias
    Eigen::Vector3d bias_w; ///< angular velocity bias
};

/// @brief Timestamped gyroscope and accelerometer measurements.
struct OdometryData {
    using Ptr = std::shared_ptr<OdometryData>;

    /// @brief Default constructor with zero measurements and identity pose.
    OdometryData()
        : t_ns(0),
          linear_velocity(Eigen::Vector3d::Zero()),
          angular_velocity(Eigen::Vector3d::Zero()),
          pose(Sophus::SE3d()) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ///< timestamp in nanoseconds
    int64_t t_ns;
    ///< Linear velocity measurement
    Eigen::Vector3d linear_velocity;
    ///< Angular velocity measurement
    Eigen::Vector3d angular_velocity;
    ///< Pose measurement
    Sophus::SE3d pose;
};

} // namespace stella_vslam

#endif // STELLA_VSLAM_ODOMETRY_TYPE_HPP
