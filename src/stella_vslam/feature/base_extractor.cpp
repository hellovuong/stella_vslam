//
// Created by vuong on 3/8/23.
//

#include "stella_vslam/feature/base_extractor.h"
namespace stella_vslam::feature {
feature_type_t base_extractor::load_feature_type(const YAML::Node& yaml_node) {
    auto feature_type_str = yaml_node["type"].as<std::string>();
    if (feature_type_str == "ORB") {
        return feature_type_t::ORB;
    }
    else if (feature_type_str == "SuperPoint") {
        return feature_type_t::SuperPoint;
    }
    throw std::runtime_error("Invalid camera model: " + feature_type_str + ". Support: ORB, SuperPoint");
}
} // namespace stella_vslam::feature