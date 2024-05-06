#include <Eigen/Geometry>

namespace stella_vslam::util {
class geometry {
public:
    static inline double get_yaw(const Eigen::Quaterniond& q) {
        double yaw;

        double sqw;
        double sqx;
        double sqy;
        double sqz;

        sqx = q.x() * q.x();
        sqy = q.y() * q.y();
        sqz = q.z() * q.z();
        sqw = q.w() * q.w();

        // Cases derived from https://orbitalstation.wordpress.com/tag/quaternion/
        double sarg = -2 * (q.x() * q.z() - q.w() * q.y()) / (sqx + sqy + sqz + sqw); /* normalization added from urdfom_headers */

        if (sarg <= -0.99999) {
            yaw = -2 * atan2(q.y(), q.x());
        }
        else if (sarg >= 0.99999) {
            yaw = 2 * atan2(q.y(), q.x());
        }
        else {
            yaw = atan2(2 * (q.x() * q.y() + q.w() * q.z()), sqw + sqx - sqy - sqz);
        }
        return normalized_angle(yaw);
    }
    static inline double normalized_angle(double angle) {
        // Ensure that the orientation is in the range [-pi, pi)
        while (angle < -M_PI) {
            angle += 2 * M_PI;
        }
        while (angle >= M_PI) {
            angle -= 2 * M_PI;
        }
        return angle;
    }
};
} // namespace stella_vslam::util
