//
// Created by vuong on 2/28/23.
//

#ifndef STELLA_VSLAM_LOOP_DETECTOR_HLOC_H
#define STELLA_VSLAM_LOOP_DETECTOR_HLOC_H

#include <memory>
#include <Eigen/Core>
#include <atomic>

#include "stella_vslam/hloc/hloc_database.h"

namespace stella_vslam::module {

class loop_detector_hloc {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    loop_detector_hloc() = default;
    /**
     * Enable loop detection
     */
    void enable_loop_detector();

    /**
     * Disable loop detection
     */
    void disable_loop_detector();

    /**
     * Get the loop detector status
     */
    bool is_enabled() const;

    /**
     * Set the current keyframe
     */
    void set_current_keyframe(const std::shared_ptr<data::keyframe>& keyfrm);

    /**
     * Detect loop candidates using Hloc
     */
    bool detect_loop_candidates();

    const std::shared_ptr<hloc::hloc_database>& getDb() const;

private:
    /**
     * called by detect_loop_candidates
     */
    bool detect_loop_candidates_impl();

    //! flag which indicates the loop detector is enabled or not
    std::atomic<bool> loop_detector_is_enabled_{true};

    //-----------------------------------------
    // variables for loop detection and correction
    //! current keyframe
    std::shared_ptr<hloc::keyframe> cur_keyfrm_ = {};

    //! hloc database
    //    hloc_database db;
    std::shared_ptr<hloc::hloc_database> db = std::make_shared<hloc::hloc_database>();
    //
    double LOOP_THRESHOLD = 0.35;
    double RELOC_THRESHOLD = 0.3;

    double PNP_INFLATION = 10.0; // amount to inflate pnp ransac sigma by (10.0 default)

    //! how many frames we should skip that are recent (avoid matching to recent frames)
    size_t RECALL_IGNORE_RECENT_COUNT = 10;

    //! last loop id
    int last_loop_count = 0;
    // viz
    std::map<int, cv::Mat> image_pool;
};

} // namespace stella_vslam::module

#endif // STELLA_VSLAM_LOOP_DETECTOR_HLOC_H
