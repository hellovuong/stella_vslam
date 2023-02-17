//
// Created by vuong on 12/27/22.
//

#ifndef STELLA_VSLAM_IO_ODOMETRY_IO_H
#define STELLA_VSLAM_IO_ODOMETRY_IO_H

#include <deque>
#include <fstream>
#include <iostream>
#include "spdlog/spdlog.h"

#include "stella_vslam/data/odometry_type.hpp"

namespace stella_vslam {
namespace data::odometry {
struct OdometryData;
}
/**
 * @brief Class read odometry data from file
 * @todo Support rosbag
 */
class odometry_io {
public:
    /**
     * Default constructor
     */
    odometry_io() = default;

    /**
     * @brief Read data odometry from file and return container
     * @tparam [in,out] Container Container store data: STL Vector, Deque, etc.
     * @param [in] file_name Data file name
     * @note Expecting line structure: \n
     * #timestamp[s] pose.position.x y z pose.orientation.x y z w twist.linear.x y z twist.angular.x y z\n
     * Unit: second, m, rad, m/s, rad/s
     * @warning Expecting file with delimiter as space ' ', however function also supports delimiter ','. Other type of delimiter is not supported
     * @return Container with data if file is exist and in correct form. Otherwise, container is empty.
     */
    template<typename Container = std::vector<data::odometry::OdometryData>>
    [[maybe_unused]] static Container read_from_file(const std::string& file_name) {
        Container result;
        std::ifstream file(file_name);
        if (not file.is_open()) {
            spdlog::warn("Can not open file. Continue without odometry data");
            return {};
        }

        std::string line;
        while (std::getline(file, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            if (line.empty() or line[0] == '#')
                continue;
            result.emplace_back(parse_line(line));
        }
        return result;
    }

    static data::odometry::OdometryData parse_line(std::string& line) {
        data::odometry::OdometryData data;
        std::stringstream ss;
        ss << line;

        // timestamp
        double ts;
        ss >> ts;

        // pose
        Eigen::Vector3d translation;
        Eigen::Quaterniond rotation;
        ss >> translation.x() >> translation.y() >> translation.z();
        ss >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.w();
        rotation.normalize();

        // linear/angular velocity
        Eigen::Vector3d linear_velocity;
        Eigen::Vector3d angular_velocity;
        ss >> linear_velocity.x() >> linear_velocity.y() >> linear_velocity.z();
        ss >> angular_velocity.x() >> angular_velocity.y() >> angular_velocity.z();

        // write data
        data.t_s = ts;
        data.pose = Sophus::SE3d(rotation, translation);
        data.linear_velocity = linear_velocity;
        data.angular_velocity = angular_velocity;

        return data;
    }

    /**
     * Get Odometry data in the interval timestamped [time_start, time_end]
     * @param odometry_data [in] Deque of odometry data
     * @param time_start [in] time to start
     * @param time_end [in] time to end
     * @param do_interpolate_front [in] Do interpolate to match the timestamp start
     * @return
     */
    [[maybe_unused]] static std::deque<data::odometry::OdometryData> get_data_interval(std::deque<data::odometry::OdometryData>& odometry_data,
                                                                                       double time_start, double time_end,
                                                                                       bool do_interpolate_front = false);
};
} // namespace stella_vslam
#endif // STELLA_VSLAM_IO_ODOMETRY_IO_H
