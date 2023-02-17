//
// Created by vuong on 12/23/22.
//

#ifndef STELLA_VSLAM_MODULE_ODOMETRY_PREINTEGRATION_HPP
#define STELLA_VSLAM_MODULE_ODOMETRY_PREINTEGRATION_HPP

#include <Eigen/Dense>
#include <utility>
#include <memory>
#include <vector>
#include <deque>
#include <shared_mutex>
#include <stella_vslam/data/odometry_type.hpp>

#include "yaml-cpp/yaml.h"

#include "stella_vslam/util/sophus_utils.hpp"

namespace stella_vslam {

namespace data::odometry {
class DeltaState;
struct OdometryData;
} // namespace data::odometry

namespace module::odometry {
/**
 * @brief Integrated pseudo-measurement that combines several consecutive odometry
 * measurements.
 * @todo Add thread-safety by mutex
 */
class IntegratedOdometryMeasurement {
public:
    typedef std::shared_ptr<IntegratedOdometryMeasurement> iomSharedPtr;
    using Vec3 = Eigen::Matrix<double, 3, 1>;

    using VecN = Eigen::Matrix<double, 6, 1>;
    using MatNN = Eigen::Matrix<double, 6, 6>;
    using MatN3 = Eigen::Matrix<double, 6, 3>;

    /**
     * Default construct
     **/
    IntegratedOdometryMeasurement() = default;

    /**
     * Construct the Integrator by yaml node
     * @param yaml_node
     */
    explicit IntegratedOdometryMeasurement(const YAML::Node& yaml_node);

    /**
     * @brief Constructor with start time and bias estimates.
     */
    IntegratedOdometryMeasurement(const Sophus::SE3d& Tbc,
                                  Eigen::Vector3d bias_v,
                                  Eigen::Vector3d bias_w,
                                  const double noise_v,
                                  const double noise_w,
                                  const double random_walk_v,
                                  const double random_walk_w,
                                  const double freq)
        : delta_t_(0),
          Tbc_(Tbc),
          Tcb_(Tbc.inverse()),
          cov_(MatNN::Zero()),
          cov_inv_computed_(false),
          d_R_d_bw_(Eigen::Matrix3d::Zero()),
          d_p_d_bw_(Eigen::Matrix3d::Zero()),
          d_p_d_bv_(Eigen::Matrix3d::Zero()),
          bias_w_(std::move(bias_w)),
          bias_v_(std::move(bias_v)),
          updated_bias_w_(Eigen::Vector3d::Zero()),
          updated_bias_v_(Eigen::Vector3d::Zero()) {
        setSigmaNoiseCov(noise_v, noise_w, random_walk_v, random_walk_w, freq);
    }

    /**
     * @brief Constructor with start time and bias estimates.
     */
    IntegratedOdometryMeasurement(const Sophus::SE3d& Tbc,
                                  Eigen::Vector3d bias_v,
                                  Eigen::Vector3d bias_w,
                                  const Eigen::DiagonalMatrix<double, 6>& sigma_noise,
                                  const Eigen::DiagonalMatrix<double, 6>& sigma_noise_random_walk)
        : delta_t_(0),
          Tbc_(Tbc),
          Tcb_(Tbc.inverse()),
          cov_(MatNN::Zero()),
          cov_inv_computed_(false),
          d_R_d_bw_(Eigen::Matrix3d::Zero()),
          d_p_d_bw_(Eigen::Matrix3d::Zero()),
          d_p_d_bv_(Eigen::Matrix3d::Zero()),
          sigma_noise_(sigma_noise),
          sigma_noise_random_walk_(sigma_noise_random_walk),
          bias_w_(std::move(bias_w)),
          bias_v_(std::move(bias_v)),
          updated_bias_w_(Eigen::Vector3d::Zero()),
          updated_bias_v_(Eigen::Vector3d::Zero()) {}

    /**
     *
     */
    void reset(const Eigen::Vector3d& bias_v, const Eigen::Vector3d& bias_w) {
        delta_t_ = (0);
        cov_ = (MatNN::Zero());
        cov_inv_computed_ = (false);
        d_R_d_bw_ = (Eigen::Matrix3d::Zero());
        d_p_d_bw_ = (Eigen::Matrix3d::Zero());
        d_p_d_bv_ = (Eigen::Matrix3d::Zero());
        sigma_noise_ = (sigma_noise_);
        sigma_noise_random_walk_ = (sigma_noise_random_walk_);
        bias_w_ = bias_w;
        bias_v_ = bias_v;
        updated_bias_w_ = (bias_w);
        updated_bias_v_ = (bias_v);
        measurements_.clear();
    }

    /// @brief Integrate new Odometry data measurements
    ///
    /// @param[in] measurements Odometry data measurements
    /// @param[in] n_v_cov diagonal of linear velocity noise covariance matrix
    /// @param[in] n_w_cov diagonal of angular velocity noise covariance matrix
    /// @note Measurements should be interpolated so that first and last measurement is at last and current frame timestamp, respectively
    void integrateMeasurements(const std::vector<data::odometry::OdometryData>& measurements);

    /// @brief Integrate new Odometry data measurements
    ///
    /// @param[in] measurements Odometry data measurements
    /// @param[in] n_v_cov diagonal of linear velocity noise covariance matrix
    /// @param[in] n_w_cov diagonal of angular velocity noise covariance matrix
    /// @note Measurements should be interpolated so that first and last measurement is at last and current frame timestamp, respectively
    void integrateMeasurements(std::deque<data::odometry::OdometryData>& measurements);

    /// @brief Integrate new Odometry data measurement
    ///
    /// @param[in] measurement Odometry data measurement
    /// @param[in] dt Delta timestamp in second
    void integrateMeasurement(const data::odometry::OdometryData& measurement, double dt);

    /**
     * @brief update preintegrated measurement
     * @details Update Delta Pose, Update Covariance, Update Jacobian wrt bias
     * @param[in] measurement: Odometry measurement
     */
    void propagateState(const data::odometry::OdometryData& measurement, double dt);

    /// @brief Predict state given this pseudo-measurement
    ///
    /// @param[in] state0 current state
    /// @param[out] state1 predicted state
    void predictState(const data::odometry::PoseBiasState& state0, data::odometry::PoseBiasState& state1);

    /// @brief Compute computeResidual between two states given this pseudo-measurement
    /// @param[in] state0 initial state
    /// @param[in] state1 next state
    /// @return computeResidual
    VecN computeResidual(const data::odometry::PoseBiasState& state0, const data::odometry::PoseBiasState& state1) const;

    /**
     * @brief Compute residual rotation
     * @param R0
     * @param R1
     * @return Tangent of residual rotation - so3
     */
    static Vec3 computeResidualRotation(const Sophus::SO3d& R0, const Sophus::SO3d& R1, const Sophus::SO3d& dR);

    /**
     * @brief Compute residual rotation
     * @param R0
     * @param R1
     * @return Residual rotation in matrix form - SO3
     */
    static Sophus::SO3d computeResidualRotationSO3(const Sophus::SO3d& R0, const Sophus::SO3d& R1, const Sophus::SO3d& dR);

    /**
     * @brief Compute Jacobian matrix for optimization
     * @param state0 [in]
     * @param state1 [in]
     * @param J_pose_1 [out] J[0] - Jacobian wrt pose 1
     * @param J_b_w [out] J[1] - Jacobian wrt bias angular
     * @param J_b_v [out] J[2] - Jacobian wrt bias linear
     * @param J_pose_2 [out] J[3] - Jacobian wrt pose 2
     * @todo Optimize computation, should we perform benchmark?
     */
    void computeJacobianOplus(const data::odometry::PoseBiasState& state0, const data::odometry::PoseBiasState& state1,
                              MatNN& J_pose_1, MatN3& J_b_w, MatN3& J_b_v, MatNN& J_pose_2);

    /**
     * @brief Merge this preintegration with previous preintegration
     * @param prev_iom_ptr
     */
    void mergePrevious(const std::shared_ptr<IntegratedOdometryMeasurement>& prev_iom_ptr);

    /// @brief Time duration of preintegrated measurement in seconds.
    double getDeltaT() const { return delta_t_; }

    /// @brief Inverse of the measurement covariance matrix
    inline const MatNN& getCovInv() const {
        if (!cov_inv_computed_) {
            cov_inv_.setIdentity();
            cov_.ldlt().solveInPlace(cov_inv_);
            cov_inv_computed_ = true;
        }
        return cov_inv_;
    }

    //! Measurement covariance matrix
    const MatNN& getCov() const { return cov_; }

    //! Get Jacobian Rotation wrt bias angular velocity
    [[maybe_unused]] const Eigen::Matrix3d& getD_R_d_bw();
    //! Get Jacobian Translation wrt bias linear velocity
    [[maybe_unused]] const Eigen::Matrix3d& getD_p_d_bw();
    //! Get Jacobian Translation wrt bias linear velocity
    [[maybe_unused]] const Eigen::Matrix3d& getD_p_d_bv();

    /**
     * @brief Set new linear/angular bias after optimization
     * @param updated_bias_v New linear velocity bias
     * @param updated_bias_w New angular velocity bias
     * @note This function have no effect on original which is cached.
     */
    void setUpdatedBias(Eigen::Vector3d& updated_bias_v, Eigen::Vector3d& updated_bias_w);

    /**
     * @brief See function setUpdatedBias()
     * @param updated_bias_v
     */
    void setUpdatedBiasV(Eigen::Vector3d& updated_bias_v);

    /**
     * @brief See function setUpdatedBias()
     * @param updated_bias_w
     */
    void setUpdatedBiasW(Eigen::Vector3d& updated_bias_w);

    /**
     * @warning Non discard return value
     * @return Updated linear velocity bias
     */
    [[nodiscard]] Eigen::Vector3d getUpdatedBiasW() const { return updated_bias_w_; };

    /**
     * @warning Non discard return value
     * @return Updated angular velocity bias
     */
    [[nodiscard]] Eigen::Vector3d getUpdatedBiasV() const { return updated_bias_v_; };

    const Sophus::SE3d& getTbc() const;
    const Sophus::SE3d& getTcb() const;
    const Eigen::DiagonalMatrix<double, 6>& getSigmaNoise() const;
    void setSigmaNoise(const Eigen::DiagonalMatrix<double, 6>& sigmaNoise);
    const Eigen::DiagonalMatrix<double, 6>& getSigmaNoiseRandomWalk() const;
    void setSigmaNoiseRandomWalk(const Eigen::DiagonalMatrix<double, 6>& sigmaNoiseRandomWalk);
    const std::vector<data::odometry::OdometryData>& getMeasurements() const;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    void setSigmaNoiseCov(double noise_v, double noise_w, double random_walk_v, double random_walk_w, double freq);

    [[nodiscard]] Eigen::Vector3d correctLinearVel(const Eigen::Vector3d& measurement) const;
    [[nodiscard]] Eigen::Vector3d correctAngularVel(const Eigen::Vector3d& measurement) const;

    /**
     * @brief Iteratively update covariance
     * @param[in] delta_R_ik
     * @param[in] corrected_linear_vel
     * @param[in] dt
     */
    void updateCovariance(const Sophus::SO3d::Transformation& delta_R_ik,
                          const Eigen::Vector3d& corrected_linear_vel,
                          double dt);

    /**
     * @brief Iteratively update jacobian state wrt bias linear/angular bias
     * @param [in] delta_R_ik
     * @param [in] corrected_linear_vel
     * @param [in] dt
     */
    void updateJacobians(const Sophus::SO3d::Transformation& delta_R_ik,
                         const Eigen::Vector3d& corrected_linear_vel,
                         double dt);

    /**
     *
     * @param bias_w_
     * @return
     */
    Sophus::SO3d getDeltaRotation(const Eigen::Vector3d& bias_w_) const;

    /**
     *
     * @param bias_v_
     * @param bias_w_
     * @return
     */
    Sophus::SE3d::TranslationMember getDeltaTranslation(const Eigen::Vector3d& bias_v_, const Eigen::Vector3d& bias_w_) const;

private:
    double delta_t_{};                         ///< Delta time in seconds
    data::odometry::DeltaState delta_state_{}; ///< Delta state, from frame i to frame j

    ///< Transformation from camera to odometry
    Sophus::SE3d Tbc_;
    Sophus::SE3d Tcb_;

    // Covariance related
    MatNN cov_;                          ///< Measurement covariance
    mutable MatNN cov_inv_;              ///< Cached inverse of measurement covariance
    mutable bool cov_inv_computed_{};    ///< If the cached inverse covariance is computed
    Sophus::Matrix6d cov_random_walk_{}; ///< Bias Random Walk covariance

    // Jacobian
    Eigen::Matrix3d d_R_d_bw_; // Jacobian of delta rotation wrt angular velocity bias

    Eigen::Matrix3d d_p_d_bw_; // Jacobian of delta translation wrt angular velocity bias
    Eigen::Matrix3d d_p_d_bv_; // Jacobian of delta translation wrt linear velocity bias

    mutable Eigen::DiagonalMatrix<double, 6> sigma_noise_;             //! diagonal matrix of noise measurement (constant)
    mutable Eigen::DiagonalMatrix<double, 6> sigma_noise_random_walk_; //! diagonal matrix of random walk model (constant)

    // Original bias when measurements were integrating
    mutable Eigen::Vector3d bias_w_; // Original Bias of angular velocity
    mutable Eigen::Vector3d bias_v_; // Original Bias of linear  velocity

    // Updated bias after optimization
    Eigen::Vector3d updated_bias_w_; // Updated Bias of angular velocity
    Eigen::Vector3d updated_bias_v_; // Updated Bias of linear  velocity

    // Saved measurement for reintegration or merge-integration
    std::vector<data::odometry::OdometryData> measurements_;

    std::mutex mutex_iom; // A mutex to ensure thread-safety
};
} // namespace module::odometry
} // namespace stella_vslam

#endif // STELLA_VSLAM_MODULE_ODOMETRY_PREINTEGRATION_HPP
