#ifndef EXAMPLE_UTIL_TUM_RGBD_UTIL_H
#define EXAMPLE_UTIL_TUM_RGBD_UTIL_H

#include <string>
#include <utility>
#include <vector>
#include "stella_vslam/type.h"

class tum_rgbd_sequence {
public:
    struct frame {
        frame(std::string rgb_img_path, std::string depth_img_path, const double timestamp,
              const Eigen::Isometry2d& robot_pose = {}, stella_vslam::Vec3_t velocity = stella_vslam::Vec3_t())
            : rgb_img_path_(std::move(rgb_img_path)), depth_img_path_(std::move(depth_img_path)), timestamp_(timestamp),
              robot_pose_(robot_pose), velocity_(std::move(velocity)){};

        const std::string rgb_img_path_;
        const std::string depth_img_path_;
        const double timestamp_;
        const Eigen::Isometry2d robot_pose_;
        const stella_vslam::Vec3_t velocity_;
    };

    explicit tum_rgbd_sequence(const std::string& seq_dir_path, double min_timediff_thr = 0.1);

    virtual ~tum_rgbd_sequence() = default;

    [[nodiscard]] std::vector<frame> get_frames() const;

    [[nodiscard]] std::vector<stella_vslam::Vec3_t> get_velocity() const;

private:
    struct img_info {
        img_info(const double timestamp, std::string img_file_path)
            : timestamp_(timestamp), img_file_path_(std::move(img_file_path)){};

        const double timestamp_;
        const std::string img_file_path_;
    };

    static std::vector<img_info> acquire_image_information(const std::string& seq_dir_path,
                                                           const std::string& timestamp_file_path);

    static std::vector<std::pair<double, stella_vslam::Vec3_t>> acquire_velocity_information(const std::string& seq_dir_path);
    static void acquire_odometry_information(const std::string& seq_dir_path, std::vector<double>& timestamps,
                                             std::vector<Eigen::Isometry2d>& poses, std::vector<stella_vslam::Vec3_t>& velocities);

    std::vector<double> timestamps_;
    std::vector<std::string> rgb_img_file_paths_;
    std::vector<std::string> depth_img_file_paths_;
    std::vector<stella_vslam::Vec3_t> robot_velocity_;
    std::vector<Eigen::Isometry2d> robot_pose_;
};
#endif // EXAMPLE_UTIL_TUM_RGBD_UTIL_H
