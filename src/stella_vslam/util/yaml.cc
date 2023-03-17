#include "stella_vslam/util/yaml.h"

namespace stella_vslam::util {

std::vector<std::vector<float>> get_rectangles(const YAML::Node& node) {
    auto rectangles = node.as<std::vector<std::vector<float>>>(std::vector<std::vector<float>>());
    for (const auto& v : rectangles) {
        if (v.size() != 4) {
            throw std::runtime_error("mask rectangle must contain four parameters");
        }
        if (v.at(0) >= v.at(1)) {
            throw std::runtime_error("x_max must be greater than x_min");
        }
        if (v.at(2) >= v.at(3)) {
            throw std::runtime_error("y_max must be greater than x_min");
        }
    }
    return rectangles;
}
hloc::hfnet_params gen_hf_params(const YAML::Node& node) {
    hloc::hfnet_params result;

    result.onnx_model_path = node["onnx_file"].as<std::string>();
    result.engine_path = node["engine_file"].as<std::string>();
    result.cache_path = node["cache_file"].as<std::string>();

    result.image_width = node["image_width"].as<int>();
    result.image_height = node["image_height"].as<int>();

    return result;
}

} // namespace stella_vslam
