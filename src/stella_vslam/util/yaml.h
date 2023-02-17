#ifndef STELLA_VSLAM_UTIL_YAML_H
#define STELLA_VSLAM_UTIL_YAML_H

#include <string>

#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include "se3.hpp"

namespace stella_vslam::util {

inline YAML::Node yaml_optional_ref(const YAML::Node& ref_node, const std::string& key) {
    return ref_node[key] ? ref_node[key] : YAML::Node();
}

std::vector<std::vector<float>> get_rectangles(const YAML::Node& node);

template<typename T>
Sophus::SE3<T> get_se3(const YAML::Node& node) {
    if (node) {
        Eigen::Matrix<T, 4, 4, Eigen::RowMajor> Mat(node.as<std::vector<T>>().data());
        Eigen::Quaternion<T> q(Mat.template block<3, 3>(0, 0));
        q.normalize();
        Eigen::Matrix<T, 3, 1> trans(Mat.template block<3, 1>(0, 3));
        return {q, trans};
    }

    return {};
}

} // namespace stella_vslam::util

#endif // STELLA_VSLAM_UTIL_YAML_H
