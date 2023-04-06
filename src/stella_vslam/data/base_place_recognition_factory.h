//
// Created by vuong on 4/5/23.
//

#ifndef STELLA_VSLAM_DATA_BASE_PLACE_RECOGNITION_FACTORY_H
#define STELLA_VSLAM_DATA_BASE_PLACE_RECOGNITION_FACTORY_H

#include <yaml-cpp/yaml.h>
#include "stella_vslam/data/base_place_recognition.h"
#include "stella_vslam/data/bow_database.h"
#include "stella_vslam/data/hf_net_database.h"
#include "spdlog/spdlog.h"
#include "stella_vslam/util/yaml.h"

namespace stella_vslam::data {
class base_place_recognition_factory {
public:
    static base_place_recognition* create(const YAML::Node& node) {
        const auto vpr_type = base_place_recognition::load_vpr_type(node);
        base_place_recognition* vpr = nullptr;
        try {
            switch (vpr_type) {
                case place_recognition_type::BoW: {
                    // load ORB vocabulary
                    auto vocab_file_path = node["bow_voc"].as<std::string>();
                    spdlog::info("loading ORB vocabulary: {}", vocab_file_path);
                    auto bow_vocab_ = data::bow_vocabulary_util::load(vocab_file_path);
                    vpr = new bow_database(bow_vocab_);
                    break;
                }
                case place_recognition_type::HF_Net: {
                    auto hf_net_ = new hloc::hf_net(util::gen_hf_params(node));
                    vpr = new hf_net_database(hf_net_);
                    break;
                }
            }
        }
        catch (const std::exception& e) {
            spdlog::debug("failed in loading place recognition parameters: {}", e.what());
            delete vpr;
            throw;
        }

        assert(vpr != nullptr);

        return vpr;
    }
};
} // namespace stella_vslam::data

#endif // STELLA_VSLAM_DATA_BASE_PLACE_RECOGNITION_FACTORY_H
