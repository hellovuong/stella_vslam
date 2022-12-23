//
// Created by vuong on 12/23/22.
//

#ifndef STELLA_VSLAM_PREINTEGRATION_HPP
#define STELLA_VSLAM_PREINTEGRATION_HPP

#include <Eigen/Dense>
#include <utility>

#include "stella_vslam/optimize/internal/odometry/odometry_type.hpp"

namespace stella_vslam {
/**
 * @brief Integrated pseudo-measurement that combines several consecutive odometry
 * measurements.
 */
class IntegratedOdometryMesurement {
public:
    using Ptr = std::shared_ptr<IntegratedOdometryMesurement>;
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using VecN = Eigen::Matrix<double, se3_SIZE, 1>;
    using MatNN = Eigen::Matrix<double, se3_SIZE, se3_SIZE>;
    using MatN3 = Eigen::Matrix<double, se3_SIZE, 3>;
    using MatN6 = Eigen::Matrix<double, se3_SIZE, 6>;

    /**
     * Default construct
     **/
    IntegratedOdometryMesurement() = default;

    /**
     * @brief Constructor with start time and bias estimates.
     */
    IntegratedOdometryMesurement(const int64_t start_t_ns,
                                 Eigen::Vector3d bias_v,
                                 Eigen::Vector3d bias_w)
        : start_t_ns(start_t_ns),
          cov_inv_computed(false),
          bias_w(std::move(bias_w)),
          bias_v(std::move(bias_v)) {
        cov.setZero();
        d_state_d_bv.setZero();
        d_state_d_bw.setZero();
    }

    /// @brief Propagate current state given OdometryData and optionally compute
    /// Jacobians.
    ///
    /// @param[in] curr_state current state
    /// @param[in] data Odometry data
    /// @param[out] next_state predicted state
    /// @param[out] d_next_d_curr Jacobian of the predicted state with respect
    /// to current state
    /// @param[out] d_next_d_v Jacobian of the predicted state with respect
    /// linear velocity measurement
    /// @param[out] d_next_d_w Jacobian of the predicted state with respect
    /// angular velocity measurement
    inline static void propagateState(const PoseState& curr_state,
                                      const OdometryData& data,
                                      PoseState& next_state,
                                      MatNN* d_next_d_curr = nullptr,
                                      MatN3* d_next_d_v = nullptr,
                                      MatN3* d_next_d_w = nullptr) {}

    /// @brief Integrate Odometry data
    ///
    /// @param[in] data Odometry data
    /// @param[in] n_v_cov diagonal of linear velocity noise covariance matrix
    /// @param[in] n_w_cov diagonal of angular velocity noise covariance matrix
    void integrate(const OdometryData& data, const Vec3& n_v_cov,
                   const Vec3& n_w_cov) {}

    /// @brief Predict state given this pseudo-measurement
    ///
    /// @param[in] state0 current state
    /// @param[out] state1 predicted state
    void predictState(const PoseState& state0, const Eigen::Vector3d& g,
                      PoseState& state1) const {}

    /// @brief Compute residual between two states given this pseudo-measurement
    /// and optionally compute Jacobians.
    ///
    /// @param[in] state0 initial state
    /// @param[in] state1 next state
    /// @param[in] curr_bv current estimate of linear velocity bias
    /// @param[in] curr_bw current estimate of angular velocity bias
    /// @param[out] d_res_d_state0 if not nullptr, Jacobian of the residual with
    /// respect to state0
    /// @param[out] d_res_d_state1 if not nullptr, Jacobian of the residual with
    /// respect to state1
    /// @param[out] d_res_d_bv if not nullptr, Jacobian of the residual with
    /// respect to linear velocity bias
    /// @param[out] d_res_d_bw if not nullptr, Jacobian of the residual with
    /// respect to angular velocity bias
    /// @return residual
    VecN residual(const PoseState& state0, const Eigen::Vector3d& g,
                  const PoseState& state1, const Eigen::Vector3d& curr_bg,
                  const Eigen::Vector3d& curr_ba, MatNN* d_res_d_state0 = nullptr,
                  MatNN* d_res_d_state1 = nullptr, MatN3* d_res_d_bv = nullptr,
                  MatN3* d_res_d_bw = nullptr) const {}

    /// @brief Time duretion of preintegrated measurement in nanoseconds.
    int64_t get_dt_ns() const { return delta_state.t_ns; }

    /// @brief Start time of preintegrated measurement in nanoseconds.
    int64_t get_start_t_ns() const { return start_t_ns; }

    /// @brief Inverse of the measurement covariance matrix
    inline const MatNN& get_cov_inv() const {
        if (!cov_inv_computed) {
            cov_inv.setIdentity();
            cov.ldlt().solveInPlace(cov_inv);
            cov_inv_computed = true;
        }
        return cov_inv;
    }

    /// @brief Measurement covariance matrix
    const MatNN& get_cov() const { return cov; }

    /// @brief Jacobian of delta state with respect to linear velocity bias
    const MatN3& get_d_state_d_bv() const { return d_state_d_bv; }

    /// @brief Jacobian of delta state with respect to angular velocity bias
    const MatN3& get_d_state_d_bw() const { return d_state_d_bw; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    ///< Integration start time in nanoseconds
    int64_t start_t_ns;

    ///< Delta state
    PoseState delta_state;

    ///< Measurement covariance
    MatNN cov;
    ///< Cached inverse of measurement covariance
    mutable MatNN cov_inv;
    ///< If the cached inverse covariance is computed
    mutable bool cov_inv_computed;

    //! Jacobian of delta state with respect to angular velocity bias
    MatN3 d_state_d_bw;
    //! Jacobian of delta state with respect to linear velocity bias
    MatN3 d_state_d_bv;

    //! Bias of angular velocity
    Eigen::Vector3d bias_w;
    //! Bias of linear  velocity
    Eigen::Vector3d bias_v;
};
} // namespace stella_vslam

#endif // STELLA_VSLAM_PREINTEGRATION_HPP
