//
// Created by vuong on 3/8/23.
//

#ifndef STELLA_VSLAM_FEATURE_FEATURE_FACTORY_H
#define STELLA_VSLAM_FEATURE_FEATURE_FACTORY_H

#include "stella_vslam/feature/base_extractor.h"
#include "stella_vslam/feature/orb_extractor.h"
#include "stella_vslam/feature/sp_extractor.h"
#include "stella_vslam/util/yaml.h"

#include "yaml-cpp/yaml.h"

namespace stella_vslam::feature {
class feature_factory {
public:
    static base_extractor* create(const YAML::Node& node) {
        const auto feature_type = base_extractor::load_feature_type(node["Feature"]);
        base_extractor* extractor = nullptr;
        try {
            switch (feature_type) {
                case feature_type_t::ORB: {
                    const auto preprocessing_params = node["Preprocessing"];
                    auto mask_rectangles = util::get_rectangles(preprocessing_params["mask_rectangles"]);
                    const auto min_size = preprocessing_params["min_size"].as<unsigned int>(800);
                    if (mask_rectangles.empty())
                        extractor = new orb_extractor(node["Feature"], min_size);
                    else
                        extractor = new orb_extractor(node["Feature"], min_size, mask_rectangles);
                    break;
                }
                case feature_type_t::SuperPoint: {
                    extractor = new sp_extractor(node["Feature"]);
                    break;
                }
            }
        }
        catch (const std::exception& e) {
            spdlog::debug("failed in loading extractor parameters: {}", e.what());
            if (extractor) {
                delete extractor;
                extractor = nullptr;
            }
            throw;
        }

        assert(extractor != nullptr);

        return extractor;
    }
};
} // namespace stella_vslam::feature
#endif // STELLA_VSLAM_FEATURE_FEATURE_FACTORY_H
