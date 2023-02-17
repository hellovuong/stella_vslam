#ifndef EXAMPLE_UTIL_TUM_RGBD_UTIL_H
#define EXAMPLE_UTIL_TUM_RGBD_UTIL_H

#include <string>
#include <utility>
#include <vector>
#include <deque>

#include "stella_vslam/data/odometry_type.hpp"

class tum_rgbd_sequence {
public:
    struct frame {
        frame(std::string rgb_img_path, std::string depth_img_path, const double timestamp)
            : rgb_img_path_(std::move(rgb_img_path)), depth_img_path_(std::move(depth_img_path)), timestamp_(timestamp){};

        const std::string rgb_img_path_;
        const std::string depth_img_path_;
        const double timestamp_;
    };

    explicit tum_rgbd_sequence(const std::string& seq_dir_path, double min_timediff_thr = 0.1);

    virtual ~tum_rgbd_sequence() = default;

    [[nodiscard]] std::vector<frame> get_frames() const;
    [[nodiscard]] const std::deque<stella_vslam::data::odometry::OdometryData>& get_odometry_data() const { return odometry_data_; };

private:
    struct img_info {
        img_info(const double timestamp, std::string img_file_path)
            : timestamp_(timestamp), img_file_path_(std::move(img_file_path)){};

        const double timestamp_;
        const std::string img_file_path_;
    };

    [[nodiscard]] static std::vector<img_info> acquire_image_information(const std::string& seq_dir_path,
                                                                         const std::string& timestamp_file_path);

    std::vector<double> timestamps_;
    std::vector<std::string> rgb_img_file_paths_;
    std::vector<std::string> depth_img_file_paths_;
    std::deque<stella_vslam::data::odometry::OdometryData> odometry_data_;
};

#endif // EXAMPLE_UTIL_TUM_RGBD_UTIL_H
