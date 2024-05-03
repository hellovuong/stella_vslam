#include "tum_rgbd_util.h"

#include <cstddef>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <utility>
#include "stella_vslam/type.h"

tum_rgbd_sequence::tum_rgbd_sequence(const std::string& seq_dir_path, const double min_timediff_thr) {
    // listing up the files in rgb/ and depth/ directories
    const auto rgb_img_infos = acquire_image_information(seq_dir_path, seq_dir_path + "/rgb.txt");
    const auto depth_img_infos = acquire_image_information(seq_dir_path, seq_dir_path + "/depth.txt");
    // const auto velocity_infos = acquire_velocity_information(seq_dir_path);

    std::vector<double> odom_timestamps;
    std::vector<Eigen::Isometry2d> odom_poses;
    std::vector<stella_vslam::Vec3_t> odom_velocities;
    acquire_odometry_information(seq_dir_path, odom_timestamps, odom_poses, odom_velocities);

    // find the nearest depth frame for each of the RGB frames
    for (const auto& rgb_img_info : rgb_img_infos) {
        // untie RGB frame information
        const auto& rgb_img_timestamp = rgb_img_info.timestamp_;
        const auto& rgb_img_file_path = rgb_img_info.img_file_path_;

        // nearest depth frame information
        auto nearest_depth_img_timestamp = depth_img_infos.begin()->timestamp_;
        auto nearest_depth_img_file_path = depth_img_infos.begin()->img_file_path_;
        double min_timediff = std::abs(rgb_img_timestamp - nearest_depth_img_timestamp);

        // calc time diff and find the nearest depth frame
        for (const auto& depth_img_info : depth_img_infos) {
            // untie RGB frame information
            const auto& depth_img_timestamp = depth_img_info.timestamp_;
            const auto& depth_img_file_path = depth_img_info.img_file_path_;
            // calc time diff
            const auto timediff = std::abs(rgb_img_timestamp - depth_img_timestamp);
            // find the nearest depth frame
            if (timediff < min_timediff) {
                min_timediff = timediff;
                nearest_depth_img_timestamp = depth_img_timestamp;
                nearest_depth_img_file_path = depth_img_file_path;
            }
        }

        // reject if the time diff is over the threshold
        if (min_timediff_thr < min_timediff) {
            continue;
        }

        timestamps_.push_back((rgb_img_timestamp + nearest_depth_img_timestamp) / 2.0);
        rgb_img_file_paths_.push_back(rgb_img_file_path);
        depth_img_file_paths_.push_back(nearest_depth_img_file_path);
    }

    if (odom_velocities.empty())
        return;

    assert(odom_timestamps.size() == odom_poses.size());
    assert(odom_timestamps.size() == odom_velocities.size());

    for (const auto& rgb_img_info : rgb_img_infos) {
        // untie RGB frame information
        const auto& rgb_img_timestamp = rgb_img_info.timestamp_;

        // nearest depth frame information
        auto nearest_odom_timestamp = odom_timestamps.at(0);
        auto nearest_vel = odom_velocities.at(0);
        auto nearest_pose = odom_poses.at(0);

        double min_timediff = std::abs(rgb_img_timestamp - nearest_odom_timestamp);

        // calc time diff and find the nearest depth frame
        for (size_t i = 0; i < odom_timestamps.size(); i++) {
            // untie RGB frame information
            const auto& odom_timestamp = odom_timestamps.at(i);
            const auto& odom_velocity = odom_velocities.at(i);
            const auto& odom_pose = odom_poses.at(i);
            // calc time diff
            const auto timediff = std::abs(rgb_img_timestamp - odom_timestamp);
            // find the nearest depth frame
            if (timediff < min_timediff) {
                min_timediff = timediff;
                nearest_odom_timestamp = odom_timestamp;
                nearest_vel = odom_velocity;
                nearest_pose = odom_pose;
            }
        }

        // reject if the time diff is over the threshold
        if (min_timediff_thr < min_timediff) {
            continue;
        }
        robot_pose_.push_back(nearest_pose);
        robot_velocity_.push_back(nearest_vel);
    }
}

std::vector<tum_rgbd_sequence::frame> tum_rgbd_sequence::get_frames() const {
    std::vector<frame> frames;
    assert(timestamps_.size() == rgb_img_file_paths_.size());
    assert(timestamps_.size() == depth_img_file_paths_.size());
    assert(rgb_img_file_paths_.size() == depth_img_file_paths_.size());
    if (not robot_velocity_.empty()) {
        assert(robot_velocity_.size() == rgb_img_file_paths_.size());
    }
    for (unsigned int i = 0; i < timestamps_.size(); ++i) {
        frames.emplace_back(rgb_img_file_paths_.at(i), depth_img_file_paths_.at(i), timestamps_.at(i),
                            !robot_pose_.empty() ? robot_pose_.at(i) : Eigen::Isometry2d(),
                            !robot_velocity_.empty() ? robot_velocity_.at(i) : stella_vslam::Vec3_t());
    }
    return frames;
}

std::vector<tum_rgbd_sequence::img_info> tum_rgbd_sequence::acquire_image_information(const std::string& seq_dir_path,
                                                                                      const std::string& timestamp_file_path) {
    std::vector<tum_rgbd_sequence::img_info> img_infos;

    // load timestamps
    std::ifstream ifs_timestamp;
    ifs_timestamp.open(timestamp_file_path.c_str());
    if (!ifs_timestamp) {
        throw std::runtime_error("Could not load a timestamp file from " + timestamp_file_path);
    }

    // load header row
    std::string s;
    //    getline(ifs_timestamp, s);
    //    getline(ifs_timestamp, s);
    //    getline(ifs_timestamp, s);

    while (!ifs_timestamp.eof()) {
        getline(ifs_timestamp, s);
        if (!s.empty() and std::strcmp(&s.front(), "#") != 0) {
            std::stringstream ss;
            ss << s;
            double timestamp;
            std::string img_file_name;
            ss >> timestamp >> img_file_name;
            img_infos.emplace_back(timestamp, seq_dir_path + "/" + img_file_name);
        }
    }

    ifs_timestamp.close();

    return img_infos;
}

std::vector<std::pair<double, stella_vslam::Vec3_t>> tum_rgbd_sequence::acquire_velocity_information(const std::string& seq_dir_path) {
    std::vector<std::pair<double, stella_vslam::Vec3_t>> results;
    auto data_file = seq_dir_path + "/odom.txt";
    // load data
    std::ifstream ifs_odom_data;
    ifs_odom_data.open(data_file.c_str());
    if (!ifs_odom_data) {
        return {};
        // throw std::runtime_error("Could not load a odom file from " + data_file);
    }

    // load header row
    std::string s;
    getline(ifs_odom_data, s);
    while (!ifs_odom_data.eof()) {
        getline(ifs_odom_data, s);
        if (!s.empty() and std::strcmp(&s.front(), "#") != 0) {
            std::stringstream ss;
            ss << s;
            double timestamp;
            double px, py, pz, qx, qy, qz, qw;
            double vx, vy, vz, wx, wy, wz;
            ss >> timestamp >> px >> py >> pz >> qx >> qy >> qz >> qw >> vx >> vy >> vz >> wx >> wy >> wz;
            results.emplace_back(timestamp, stella_vslam::Vec3_t{vx, vy, wz});
        }
    }

    ifs_odom_data.close();

    return results;
}

void tum_rgbd_sequence::acquire_odometry_information(const std::string& seq_dir_path, std::vector<double>& timestamps,
                                                     std::vector<Eigen::Isometry2d>& poses, std::vector<stella_vslam::Vec3_t>& velocities) {
    auto data_file = seq_dir_path + "/odom.txt";
    // load data
    std::ifstream ifs_odom_data;
    ifs_odom_data.open(data_file.c_str());
    if (!ifs_odom_data) {
        return;
    }

    // load header row
    std::string s;
    getline(ifs_odom_data, s);
    while (!ifs_odom_data.eof()) {
        getline(ifs_odom_data, s);
        if (!s.empty() and std::strcmp(&s.front(), "#") != 0) {
            std::stringstream ss;
            ss << s;
            double timestamp;
            double px, py, pz, qx, qy, qz, qw;
            double vx, vy, vz, wx, wy, wz;
            ss >> timestamp >> px >> py >> pz >> qx >> qy >> qz >> qw >> vx >> vy >> vz >> wx >> wy >> wz;
            timestamps.push_back(timestamp);
            Eigen::Matrix2d Rot2d;
            // clang-format off
            Rot2d << qz, -qw,
                     qw,  qz;
            // clang-format on
            Eigen::Vector2d Trans2d;
            Trans2d << px, py;
            Eigen::Isometry2d odom_pose;
            odom_pose.linear() = Rot2d;
            odom_pose.translation() = Trans2d;
            poses.push_back(odom_pose);
            velocities.emplace_back(vx, vy, wz);
        }
    }
    ifs_odom_data.close();
}
