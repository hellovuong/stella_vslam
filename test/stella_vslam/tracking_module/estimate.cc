//
// Created by vuong on 4/21/23.
//

#include "gtest/gtest.h"
#include "stella_vslam/tracking_module.h"
#include <Eigen/Core>

TEST(robot_to_cam_twist, returns_expected_result) {
    Eigen::Matrix4d Tbc;
    // clang-format off
    Tbc <<  0.0, -1.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 1.0, 3.0,
            0.0, 0.0, 0.0, 1.0;
    // clang-format on
    Eigen::Vector3d robot_vel(1.0, 2.0, 3.0);
    double ts = 0.1;

    Eigen::Matrix4d result = stella_vslam::tracking_module::robot_to_cam_twist(Tbc, robot_vel, ts);
    std::cout << result << std::endl;
    Eigen::Matrix4d expected_result;
    // clang-format off
    expected_result <<  0.0, -1.0, 0.0, -0.1,
                        1.0, 0.0, 0.0, 0.2,
                        0.0, 0.0, 1.0, 2.9,
                        0.0, 0.0, 0.0, 1.0;
    // clang-format on
    ASSERT_TRUE(result.isApprox(expected_result));
}