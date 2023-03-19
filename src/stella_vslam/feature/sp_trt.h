//
// Created by vuong on 3/7/23.
//

#ifndef STELLA_VSLAM_FEATURE_SP_TRT_H
#define STELLA_VSLAM_FEATURE_SP_TRT_H

#include <vector>
#include <string>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>

#include "common_trt/buffers.h"
#include "common_trt/common.h"
#include "common_trt/logger.h"

#include <spdlog/spdlog.h>

using namespace samplesCommon;
using namespace sample;


namespace stella_vslam::feature {

struct sp_params {
    int max_keypoints;
    double keypoint_threshold;
    int remove_borders;
    int dla_core;
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    std::string onnx_file;
    std::string engine_file;
};

class sp_trt {
public:
    explicit sp_trt(sp_params super_point_config);

    bool build();

    bool infer(const cv::Mat& image);

    [[maybe_unused]] void visualization(const std::string& image_name, const cv::Mat& image);
    [[maybe_unused]] static void visualization(const std::vector<cv::KeyPoint>& kps,
                       const std::string& image_name, const cv::Mat& image);
    void save_engine();

    bool deserialize_engine();

    std::vector<cv::KeyPoint> detect(const cv::Mat& image);
    std::vector<cv::KeyPoint> detect(const cv::Mat& image, int max_keypoint);
    cv::Mat compute(const std::vector<cv::KeyPoint>& kps);
    void detect_and_compute(const cv::Mat& image, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, int max_kpts = 0);

private:
    sp_params super_point_config_;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims semi_dims_{};
    nvinfer1::Dims desc_dims_{};
    std::unique_ptr<BufferManager> buffer_manager_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::vector<int>> keypoints_;
    std::vector<std::vector<double>> descriptors_;

    bool construct_network(
        TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
        TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TensorRTUniquePtr<nvonnxparser::IParser>& parser) const;

    bool process_input(const BufferManager& buffers, const cv::Mat& image);

    bool process_input(const std::unique_ptr<BufferManager>& buffers,
                       const cv::Mat& image);

    bool process_output(const BufferManager& buffers,
                        Eigen::Matrix<double, 259, Eigen::Dynamic>& features);

    bool process_output(const std::unique_ptr<BufferManager>& buffers,
                        Eigen::Matrix<double, 259, Eigen::Dynamic>& features);

    static void remove_borders(std::vector<std::vector<int>>& keypoints,
                        std::vector<float>& scores, int border, int height,
                        int width);

    static std::vector<size_t> sort_indexes(std::vector<float>& data);

    static void top_k_keypoints(std::vector<std::vector<int>>& keypoints,
                         std::vector<float>& scores, int k);

    static void find_high_score_index(std::vector<float>& scores,
                                      std::vector<std::vector<int>>& keypoints,
                                      int h, int w, double threshold);

    static void sample_descriptors(std::vector<std::vector<int>>& keypoints,
                            float* descriptors,
                            std::vector<std::vector<double>>& dest_descriptors,
                            int dim, int h, int w, int s = 8);
};
typedef std::shared_ptr<stella_vslam::feature::sp_trt> spPtr;
} // namespace stella_vslam::feature

#endif // STELLA_VSLAM_FEATURE_SP_TRT_H
