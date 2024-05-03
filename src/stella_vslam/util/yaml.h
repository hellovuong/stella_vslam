#ifndef STELLA_VSLAM_UTIL_YAML_H
#define STELLA_VSLAM_UTIL_YAML_H

#include <string>

#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include "stella_vslam/hloc/hf_net.h"
#include "stella_vslam/type.h"

namespace stella_vslam::util {

inline YAML::Node yaml_optional_ref(const YAML::Node& ref_node, const std::string& key) {
    return ref_node[key] ? ref_node[key] : YAML::Node();
}

std::vector<std::vector<float>> get_rectangles(const YAML::Node& node);

hloc::hfnet_params gen_hf_params(const YAML::Node& node);

stella_vslam::Mat44_t get_Mat44(const YAML::Node& node);

} // namespace stella_vslam::util

#endif // STELLA_VSLAM_UTIL_YAML_H
