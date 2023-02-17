//
// Created by vuong on 1/16/23.
//

#include "bias_random_walk_edge.h"

namespace stella_vslam::optimize::internal::odometry {
bool linear_vel_bias_random_walk_edge::read(std::istream& is) {
    for (unsigned int i = 0; i < 3; ++i) {
        is >> _measurement(i);
    }
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
bool linear_vel_bias_random_walk_edge::write(std::ostream& os) const {
    for (unsigned int i = 0; i < measurement().size(); ++i) {
        os << measurement()(i) << " ";
    }
    for (int i = 0; i < information().rows(); ++i) {
        for (int j = i; j < information().cols(); ++j) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}
void linear_vel_bias_random_walk_edge::computeError() {
    const auto* V0 = static_cast<const vertex_linear_vel_bias*>(_vertices[0]);
    const auto* V1 = static_cast<const vertex_linear_vel_bias*>(_vertices[1]);
    _error = V1->estimate() - V0->estimate();
}
void linear_vel_bias_random_walk_edge::linearizeOplus() {
    _jacobianOplusXi = -Eigen::Matrix3d::Identity();
    _jacobianOplusXj.setIdentity();
}

bool angular_vel_bias_random_walk_edge::read(std::istream& is) {
    for (unsigned int i = 0; i < 3; ++i) {
        is >> _measurement(i);
    }
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
bool angular_vel_bias_random_walk_edge::write(std::ostream& os) const {
    for (unsigned int i = 0; i < measurement().size(); ++i) {
        os << measurement()(i) << " ";
    }
    for (int i = 0; i < information().rows(); ++i) {
        for (int j = i; j < information().cols(); ++j) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}
void angular_vel_bias_random_walk_edge::computeError() {
    const auto* V0 = static_cast<const vertex_angular_vel_bias*>(_vertices[0]);
    const auto* V1 = static_cast<const vertex_angular_vel_bias*>(_vertices[1]);
    _error = V1->estimate() - V0->estimate();
}
void angular_vel_bias_random_walk_edge::linearizeOplus() {
    _jacobianOplusXi = -Eigen::Matrix3d::Identity();
    _jacobianOplusXj.setIdentity();
}
} // namespace stella_vslam::optimize::internal::odometry