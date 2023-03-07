//
// Created by vuong on 2/28/23.
//

#ifndef STELLA_VSLAM_HLOC_H
#define STELLA_VSLAM_HLOC_H

#include <torch/script.h>
#include <opencv2/core.hpp>

#include "spdlog/spdlog.h"

namespace stella_vslam {

#define SuperPointPath "/home/vuong/Dev/prv/ws/openvins_ws/src/ov_hloc/support_files/weights/models/SuperPoint_1024.pt"
#define NetVLADPath "/home/vuong/Dev/prv/ws/openvins_ws/src/ov_hloc/support_files/weights/models/NetVLAD.pt"
#define SuperGluePath "/home/vuong/Dev/prv/ws/openvins_ws/src/ov_hloc/support_files/weights/models/NetVLAD.pt"
#define UltraPointPath "/home/vuong/Dev/prv/ws/openvins_ws/src/ov_hloc/support_files/weights/models/UltraPoint.pt"

class SuperPoint {
public:
    static SuperPoint& self();
    static void Extract(
        const cv::Mat& image,
        std::vector<cv::KeyPoint>& kpts,
        cv::Mat& desc);

private:
    torch::jit::script::Module model;
    SuperPoint();
    void IExtract(
        const cv::Mat& image,
        std::vector<cv::KeyPoint>& kpts,
        cv::Mat& desc);
};

class NetVLAD {
public:
    static NetVLAD& self();
    static void Extract(
        const cv::Mat& image,
        cv::Mat& desc);

private:
    torch::jit::script::Module model;
    NetVLAD();
    void IExtract(
        const cv::Mat& image,
        cv::Mat& desc);
};

class SuperGlue {
public:
    static SuperGlue& self();
    static void Match(
        std::vector<cv::Point2f>& kpts0,
        std::vector<float>& scrs0,
        cv::Mat& desc0,
        int height0, int width0,
        std::vector<cv::Point2f>& kpts1,
        std::vector<float>& scrs1,
        cv::Mat& desc1,
        int height1, int width1,
        std::vector<int>& match_index,
        std::vector<float>& match_score);

private:
    torch::jit::script::Module model;
    SuperGlue();
    void IMatch(
        std::vector<cv::Point2f>& kpts0,
        std::vector<float>& scrs0,
        cv::Mat& desc0,
        int height0, int width0,
        std::vector<cv::Point2f>& kpts1,
        std::vector<float>& scrs1,
        cv::Mat& desc1,
        int height1, int width1,
        std::vector<int>& match_index,
        std::vector<float>& match_score);
};

class UltraPoint {
public:
    static UltraPoint& self();
    static void Extract(
        const cv::Mat& image,
        std::vector<cv::Point2f>& kpts,
        std::vector<float>& scrs,
        cv::Mat& desc);

private:
    torch::jit::script::Module model;
    UltraPoint();
    void IExtract(
        const cv::Mat& image,
        std::vector<cv::Point2f>& kpts,
        std::vector<float>& scrs,
        cv::Mat& desc);
};
} // namespace stella_vslam

#endif // STELLA_VSLAM_HLOC_H
