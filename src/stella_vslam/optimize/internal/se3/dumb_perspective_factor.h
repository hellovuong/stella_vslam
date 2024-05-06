#ifndef STELLA_VSLAM_OPTIMIZE_G2O_INTERNAL_SE3_DUMB_PERSPECTIVE_FACTOR_H
#define STELLA_VSLAM_OPTIMIZE_G2O_INTERNAL_SE3_DUMB_PERSPECTIVE_FACTOR_H
#include "stella_vslam/camera/perspective.h"
#include "stella_vslam/camera/fisheye.h"
#include "stella_vslam/camera/equirectangular.h"
#include "stella_vslam/camera/radial_division.h"
#include "stella_vslam/optimize/internal/landmark_vertex.h"
#include "stella_vslam/optimize/internal/se3/perspective_reproj_edge.h"
#include "stella_vslam/optimize/internal/se3/equirectangular_reproj_edge.h"

#include <g2o/core/robust_kernel_impl.h>

#include <cassert>
#include <memory>

namespace stella_vslam {

namespace data {
class landmark;
} // namespace data

namespace optimize::internal::se3 {
class dumb_perspective_factor_wrapper {
public:
    dumb_perspective_factor_wrapper() = delete;

    ~dumb_perspective_factor_wrapper() = default;

    dumb_perspective_factor_wrapper(const camera::base* camera, shot_vertex* shot_vtx, landmark_vertex* lm_vtx,
                                    unsigned int idx, float obs_x, float obs_y, float obs_x_right,
                                    float inv_sigma_sq, float sqrt_chi_sq, bool use_huber);

    [[nodiscard]] bool is_inlier() const;

    [[nodiscard]] bool is_outlier() const;

    void set_as_inlier() const;

    void set_as_outlier() const;

    [[nodiscard]] bool depth_is_positive() const;

    g2o::OptimizableGraph::Edge* edge_;

    const camera::base* camera_;
    const unsigned int idx_;
    const bool is_monocular_;
};

inline dumb_perspective_factor_wrapper::dumb_perspective_factor_wrapper(const camera::base* camera, shot_vertex* shot_vtx, landmark_vertex* lm_vtx,
                                                                        const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                                                                        const float inv_sigma_sq, const float sqrt_chi_sq, const bool use_huber)
    : camera_(camera), idx_(idx), is_monocular_(obs_x_right < 0) {
    // 拘束条件を設定
    switch (camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            const auto c = static_cast<const camera::perspective*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_reproj_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->setVertex(0, lm_vtx);
                edge->setVertex(1, shot_vtx);
                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_reproj_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->setVertex(0, lm_vtx);
                edge->setVertex(1, shot_vtx);

                edge_ = edge;
            }
            break;
        }
        case camera::model_type_t::Fisheye: {
            assert(false);
        }
        case camera::model_type_t::Equirectangular: {
            assert(false);
        }
        case camera::model_type_t::RadialDivision: {
            assert(false);
        } break;
    }

    // loss functionを設定
    if (use_huber) {
        auto huber_kernel = new g2o::RobustKernelHuber();
        huber_kernel->setDelta(sqrt_chi_sq);
        edge_->setRobustKernel(huber_kernel);
    }
}
[[nodiscard]] bool dumb_perspective_factor_wrapper::is_inlier() const {
    return edge_->level() == 0;
}

[[nodiscard]] bool dumb_perspective_factor_wrapper::is_outlier() const {
    return edge_->level() != 0;
}

void dumb_perspective_factor_wrapper::set_as_inlier() const {
    edge_->setLevel(0);
}

void dumb_perspective_factor_wrapper::set_as_outlier() const {
    edge_->setLevel(1);
}

} // namespace optimize::internal::se3
} // namespace stella_vslam
#endif // !STELLA_VSLAM_OPTIMIZE_G2O_INTERNAL_SE3_DUMB_PERSPECTIVE_FACTOR_H
