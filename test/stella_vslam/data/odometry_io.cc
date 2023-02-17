//
// Created by vuong on 12/27/22.
//
#include "gtest/gtest.h"

#include "stella_vslam/data/odometry_type.hpp"
#include "stella_vslam/io/odometry_io.h"

using namespace stella_vslam;

TEST(odometry_io, read_from_file) {
    std::string file_path = std::string(TEST_DATA_DIR) + "odom.txt";
    auto vec_data = odometry_io::read_from_file(file_path);
    auto deque_data = odometry_io::read_from_file<std::deque<OdometryData>>(file_path);

    auto data = vec_data.at(0);
    EXPECT_GT(data.t_s, 0);
    EXPECT_GT(vec_data.size(), 0);
    EXPECT_GT(deque_data.size(), 0);

    auto failed_file_path = file_path.replace(file_path.begin(), file_path.end(), "odom", "");
    auto failed_vec_data = odometry_io::read_from_file(failed_file_path);
    auto failed_deque_data = odometry_io::read_from_file<std::deque<OdometryData>>(failed_file_path);
    EXPECT_TRUE(failed_vec_data.empty());
    EXPECT_TRUE(failed_deque_data.empty());
}
