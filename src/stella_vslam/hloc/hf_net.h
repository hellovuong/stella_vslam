//
// Created by vuong on 3/14/23.
//

#ifndef STELLA_VSLAM_HLOC_HF_NET_H
#define STELLA_VSLAM_HLOC_HF_NET_H

#include <string>

#include <opencv2/core/matx.hpp>

#include <NvInferRuntime.h>

#include "stella_vslam/feature/common_trt/buffers.h"
#include "stella_vslam/feature/common_trt/common.h"
#include "stella_vslam/feature/common_trt/logger.h"

namespace stella_vslam::hloc {

class hf_net {

public:
    hf_net(const std::string &strModelDir, const cv::Vec4i inputShape);
    ~hf_net() = default;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold);

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                int nKeypointsNum, float threshold);

    bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr;

protected:

    bool LoadHFNetTRModel(void);

    void LoadTimingCacheFile(const std::string& strFileName, std::unique_ptr<nvinfer1::IBuilderConfig>& config, std::unique_ptr<nvinfer1::ITimingCache>& timingCache);

    void UpdateTimingCacheFile(const std::string& strFileName, std::unique_ptr<nvinfer1::IBuilderConfig>& config, std::unique_ptr<nvinfer1::ITimingCache>& timingCache);

    std::string DecideEigenFileName(const std::string& strEngineSaveDir, const nvinfer1::Dims4 inputShape);

    bool SaveEngineToFile(const std::string& strEngineSaveFile, const std::unique_ptr<nvinfer1::IHostMemory>& serializedEngine);

    bool LoadEngineFromFile(const std::string& strEngineSaveFile);

    void PrintInputAndOutputsInfo(std::unique_ptr<nvinfer1::INetworkDefinition>& network);

    bool Run(void);

    void GetLocalFeaturesFromTensor(const RTTensor &tScoreDense, const RTTensor &tDescriptorsMap,
                                    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                                    int nKeypointsNum, float threshold);

    void GetGlobalDescriptorFromTensor(const RTTensor &tDescriptors, cv::Mat &globalDescriptors);

    void Mat2Tensor(const cv::Mat &mat, RTTensor &tensor);

    void Tensor2Mat(const RTTensor &tensor, cv::Mat &mat);

    void ResamplerRT(const RTTensor &data, const cv::Mat &warp, cv::Mat &output);

    nvinfer1::Dims4 mInputShape;
    std::string mStrTRModelDir;
    std::string mStrONNXFile;
    std::string mStrCacheFile;
    bool mbVaild = false;
    std::unique_ptr<samplesCommon::BufferManager> mpBuffers;

    std::vector<RTTensor> mvInputTensors;
    nvinfer1::Dims input_dims_{};
    std::vector<RTTensor> mvOutputTensors;
    nvinfer1::Dims output_dims_{};
    std::shared_ptr<nvinfer1::IExecutionContext> mContext = nullptr;
};

} // namespace stella_vslam

#endif // STELLA_VSLAM_HLOC_HF_NET_H
