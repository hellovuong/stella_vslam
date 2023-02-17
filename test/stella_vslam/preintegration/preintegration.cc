//
// Created by vuong on 12/28/22.
//

#include "gtest/gtest.h"

#include "yaml-cpp/yaml.h"

#include "stella_vslam/module/odometry/preintegration.hpp"

using namespace stella_vslam::module::odometry;
TEST(preintegration, read_from_yaml) {
    std::string file_path = std::string(TEST_DATA_DIR) + "openloris_rgbd.yaml";
    auto yaml = YAML::LoadFile(file_path);

    EXPECT_TRUE(yaml["Odometry"]);

    auto iom_ptr = std::make_shared<IntegratedOdometryMeasurement>(yaml["Odometry"]);
    auto Tbc = iom_ptr->getTbc();

    EXPECT_NEAR(Tbc.translation().x(), 0.1, std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(Tbc.translation().y(), 0.2, std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(Tbc.translation().z(), 0.3, std::numeric_limits<double>::epsilon());

    EXPECT_NEAR(Tbc.unit_quaternion().x(), 0.0, std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(Tbc.unit_quaternion().y(), 0.0, std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(Tbc.unit_quaternion().z(), 0.0, std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(Tbc.unit_quaternion().w(), 1.0, std::numeric_limits<double>::epsilon());
}