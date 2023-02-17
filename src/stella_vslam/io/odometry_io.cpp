//
// Created by vuong on 12/27/22.
//

#include "odometry_io.h"
namespace stella_vslam {
[[maybe_unused]] std::deque<data::odometry::OdometryData> stella_vslam::odometry_io::get_data_interval(std::deque<data::odometry::OdometryData>& odometry_data,
                                                                                                       const double time_start, const double time_end,
                                                                                                       bool do_interpolate_front) {
    std::deque<data::odometry::OdometryData> result;

    if (odometry_data.empty())
        return {};

    bool interpolate_back = true;
    //    spdlog::debug("interval [{:.{}f}, {:.{}f}]", time_start, 6, time_end, 6);
    // if still inside the interval [start, end]
    while (odometry_data.front().t_s <= time_end) {
        if (odometry_data.front().t_s < time_start) {
            odometry_data.pop_front();
            continue;
        }
        if (do_interpolate_front and odometry_data.front().t_s == time_start)
            do_interpolate_front = false;

        if (interpolate_back and odometry_data.front().t_s == time_end)
            interpolate_back = false;

        result.push_back(odometry_data.front());
        odometry_data.pop_front();
    }

    if (result.empty())
        return {};

    if (do_interpolate_front) {
        result.emplace_front(time_start, odometry_data.front().pose,
                             odometry_data.front().linear_velocity,
                             odometry_data.front().angular_velocity);
    }
    if (interpolate_back) {
        result.emplace_back(time_end, odometry_data.back().pose,
                            odometry_data.back().linear_velocity,
                            odometry_data.back().angular_velocity);
    }
    result.shrink_to_fit();
    return result;
}
} // namespace stella_vslam