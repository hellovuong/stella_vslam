//
// Created by vuong on 3/18/23.
//

#include "stella_vslam/data/keyframe.h"

#include "base_place_recognition.h"
#include "spdlog/spdlog.h"

namespace stella_vslam::data {
base_place_recognition::~base_place_recognition() {
    spdlog::debug("DESTRUCT: data::base_place_recognition");
}
place_recognition_t base_place_recognition::load_vpr_type(const YAML::Node& yaml_node) {
    auto vpr_type_str = yaml_node["type"].as<std::string>();
    if (vpr_type_str == "BoW") {
        return place_recognition_t::BoW;
    }
    else if (vpr_type_str == "HF_Net") {
        return place_recognition_t::HF_Net;
    }
    throw std::runtime_error("Invalid feature type: " + vpr_type_str + ". Support: BoW, HF_Net");
}
} // namespace stella_vslam::data