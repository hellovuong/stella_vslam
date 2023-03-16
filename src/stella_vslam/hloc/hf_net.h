//
// Created by vuong on 3/14/23.
//

#ifndef STELLA_VSLAM_HLOC_HF_NET_H
#define STELLA_VSLAM_HLOC_HF_NET_H

#include <string>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include "stella_vslam/feature/common_trt/buffers.h"
#include "stella_vslam/feature/common_trt/common.h"
#include "stella_vslam/feature/common_trt/logger.h"
#include "spdlog/spdlog.h"

namespace stella_vslam::hloc {

class RTTensor {
public:
    RTTensor(void* d, nvinfer1::Dims s)
        : data(d), shape(s) {}
    void* data;
    nvinfer1::Dims shape;
};

struct hfnet_params {
    std::string onnx_model_path;
    std::string cache_path;
    std::string engine_path;
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    int image_width;
    int image_height;
};

class hf_net {
public:
    hf_net(const hfnet_params& params);

    ~hf_net() = default;

    bool Detect(const cv::Mat& image, std::vector<cv::KeyPoint>& vKeyPoints, cv::Mat& localDescriptors, cv::Mat& globalDescriptors,
                int nKeypointsNum, float threshold);

    bool Detect(const cv::Mat& image, std::vector<cv::KeyPoint>& vKeyPoints, cv::Mat& localDescriptors,
                int nKeypointsNum, float threshold);

    bool Detect(const cv::Mat& intermediate, cv::Mat& globalDescriptors);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr;

protected:
    bool LoadHFNetTRModel();

    bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                           TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                           TensorRTUniquePtr<nvonnxparser::IParser>& parser) const;

    void LoadTimingCacheFile(const std::string& strFileName, std::unique_ptr<nvinfer1::IBuilderConfig>& config, std::unique_ptr<nvinfer1::ITimingCache>& timingCache);

    void UpdateTimingCacheFile(const std::string& strFileName, std::unique_ptr<nvinfer1::IBuilderConfig>& config, std::unique_ptr<nvinfer1::ITimingCache>& timingCache);

    std::string DecideEigenFileName(const std::string& strEngineSaveDir, const nvinfer1::Dims4 inputShape);

    bool SaveEngineToFile(const std::string& strEngineSaveFile, const std::unique_ptr<nvinfer1::IHostMemory>& serializedEngine);

    bool LoadEngineFromFile(const std::string& strEngineSaveFile);

    void PrintInputAndOutputsInfo(std::unique_ptr<nvinfer1::INetworkDefinition>& network);

    bool Run(void);

    void GetLocalFeaturesFromTensor(const RTTensor& tScoreDense, const RTTensor& tDescriptorsMap,
                                    std::vector<cv::KeyPoint>& vKeyPoints, cv::Mat& localDescriptors,
                                    int nKeypointsNum, float threshold);

    void GetGlobalDescriptorFromTensor(const RTTensor& tDescriptors, cv::Mat& globalDescriptors);

    void Mat2Tensor(const cv::Mat& mat, RTTensor& tensor);

    void Tensor2Mat(const RTTensor& tensor, cv::Mat& mat);

    void ResamplerRT(const RTTensor& data, const cv::Mat& warp, cv::Mat& output);

    nvinfer1::Dims4 mInputShape;

    std::string mStrONNXFile{};
    std::string mStrCacheFile{};
    std::string mStrEngineFile{};

    bool mbVaild = false;
    std::unique_ptr<samplesCommon::BufferManager> mpBuffers = nullptr;

    std::vector<RTTensor> mvInputTensors{};
    std::vector<RTTensor> mvOutputTensors{};
    std::shared_ptr<nvinfer1::IExecutionContext> mContext = nullptr;
};

} // namespace stella_vslam::hloc

#endif // STELLA_VSLAM_HLOC_HF_NET_H
