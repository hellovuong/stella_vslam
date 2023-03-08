//
// Created by vuong on 2/28/23.
//

#ifndef STELLA_VSLAM_HLOC_HLOC_DATABASE_H
#define STELLA_VSLAM_HLOC_HLOC_DATABASE_H

#include <opencv2/core.hpp>
#include <map>

#include "stella_vslam/data/keyframe.h"
//#include "stella_vslam/hloc/hloc.h"
#include <spdlog/spdlog.h>

namespace stella_vslam::hloc {
class keyframe {
public:
    explicit keyframe(std::shared_ptr<data::keyframe> cur_keyfrm);
    //! current keyframe
    std::shared_ptr<data::keyframe> keyfrm_{};
    void computeNew();
    cv::Mat global_descriptors_{};
};
typedef std::list<std::pair<std::shared_ptr<stella_vslam::hloc::keyframe>, double>> query_res_t;

class hloc_database {
public:
    void add(const std::shared_ptr<keyframe>& keyfrm, cv::Mat& new_global_descriptor) {
        database[keyfrm] = new_global_descriptor;
    }

    query_res_t query(cv::Mat& global_descriptor) {
        query_res_t ret;
        for (auto const& desc : database) {
            auto key_frame = desc.first;
            if ((not key_frame)
                or (not key_frame->keyfrm_)
                or (key_frame->keyfrm_->will_be_erased())) {
                continue;
            }
            auto score = global_descriptor.dot(desc.second);
            if (score < 0.7)
                continue;
            ret.push_back(std::make_pair(key_frame, score));
        }
        ret.sort(comp_first);
        return ret;
    }

private:
    std::unordered_map<std::shared_ptr<keyframe>, cv::Mat> database{};
    static bool comp_first(const std::pair<std::shared_ptr<keyframe>, double>& a,
                           const std::pair<std::shared_ptr<keyframe>, double>& b) {
        return a.first > b.first;
    }
};
} // namespace stella_vslam::hloc
#endif // STELLA_VSLAM_HLOC_HLOC_DATABASE_H
