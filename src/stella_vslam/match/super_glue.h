//
// Created by vuong on 4/6/23.
//

#ifndef STELLA_VSLAM_MATCH_SUPER_GLUE_H
#define STELLA_VSLAM_MATCH_SUPER_GLUE_H

#include <string>
#include <vector>
#include <memory>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>

#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>

#include <yaml-cpp/yaml.h>

#include "stella_vslam/feature/common_trt/buffers.h"
#include "stella_vslam/feature/common_trt/logger.h"
#include "stella_vslam/feature/common_trt/logging.h"

namespace stella_vslam::match {

struct SuperGlueConfig {
    int image_width;
    int image_height;
    int dla_core;
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    std::string onnx_file;
    std::string engine_file;
};

class SuperGlue {
public:
    explicit SuperGlue(SuperGlueConfig superglue_config);

    bool build();

    bool infer(const std::vector<cv::KeyPoint>& kps0, const cv::Mat& desc0,
               const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1,
               Eigen::VectorXi& indices0,
               Eigen::VectorXi& indices1,
               Eigen::VectorXd& mscores0,
               Eigen::VectorXd& mscores1);

    void save_engine();

    bool deserialize_engine();

    static void gen_configs(const YAML::Node& superglue_node, SuperGlueConfig& superglue_config);

    int MatchingPoints(const std::vector<cv::KeyPoint>& kps0, const cv::Mat& desc0,
                       const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1,
                       std::vector<cv::DMatch>& matches);

private:
    SuperGlueConfig superglue_config_;
    std::vector<int> indices0_;
    std::vector<int> indices1_;
    std::vector<double> mscores0_;
    std::vector<double> mscores1_;

    nvinfer1::Dims keypoints_0_dims_{};
    nvinfer1::Dims scores_0_dims_{};
    nvinfer1::Dims descriptors_0_dims_{};
    nvinfer1::Dims keypoints_1_dims_{};
    nvinfer1::Dims scores_1_dims_{};
    nvinfer1::Dims descriptors_1_dims_{};
    nvinfer1::Dims output_scores_dims_{};

    std::unique_ptr<samplesCommon::BufferManager> buffers_manager_;

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                           TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                           TensorRTUniquePtr<nvonnxparser::IParser>& parser) const;

    bool process_input(const std::unique_ptr<samplesCommon::BufferManager>& buffers,
                       const std::vector<cv::KeyPoint>& kps0, const cv::Mat& desc0,
                       const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1);

    bool process_output(const std::unique_ptr<samplesCommon::BufferManager>& buffers,
                        Eigen::VectorXi& indices0,
                        Eigen::VectorXi& indices1,
                        Eigen::VectorXd& mscores0,
                        Eigen::VectorXd& mscores1);

    static std::vector<cv::KeyPoint> NormalizeKeypoints(const std::vector<cv::KeyPoint>& features, int width, int height) {
        std::vector<cv::KeyPoint> norm_features;

        for (const auto& feature : features) {
            cv::KeyPoint norm_kp;
            norm_kp.response = feature.response;
            norm_kp.pt.x = (feature.pt.x - (float)width / 2) / ((float)std::max(width, height) * 0.7f);
            norm_kp.pt.y = (feature.pt.y - (float)height / 2) / ((float)std::max(width, height) * 0.7f);
            norm_features.push_back(norm_kp);
        }
        return norm_features;
    }
};
} // namespace stella_vslam::match
#endif // STELLA_VSLAM_MATCH_SUPER_GLUE_H
