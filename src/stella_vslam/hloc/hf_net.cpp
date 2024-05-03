//
// Created by vuong on 3/14/23.
//

#include <opencv2/imgproc.hpp>
#include "hf_net.h"

namespace stella_vslam::hloc {

hf_net::hf_net(const hfnet_params& params) {
    mStrONNXFile = params.onnx_model_path;
    mStrCacheFile = params.cache_path;
    mStrEngineFile = params.engine_path;
    mInputShape = {1, params.image_height, params.image_width, 1};

    sample::setReportableSeverity(sample::Logger::Severity::kVERBOSE);
    if (not LoadHFNetTRModel()) {
        spdlog::error("Failed to construct HF_NET!");
        return;
    }

    mpBuffers = std::make_unique<samplesCommon::BufferManager>(mEngine);

    input_tensors.emplace_back(mpBuffers->getHostBuffer("image:0"), mEngine->getBindingDimensions(mEngine->getBindingIndex("image:0")));

    output_tensors.emplace_back(mpBuffers->getHostBuffer("scores_dense_nms:0"), mEngine->getBindingDimensions(mEngine->getBindingIndex("scores_dense_nms:0")));
    output_tensors.emplace_back(mpBuffers->getHostBuffer("local_descriptor_map:0"), mEngine->getBindingDimensions(mEngine->getBindingIndex("local_descriptor_map:0")));
    output_tensors.emplace_back(mpBuffers->getHostBuffer("global_descriptor:0"), mEngine->getBindingDimensions(mEngine->getBindingIndex("global_descriptor:0")));

    spdlog::info("Constructed HF_NET!");
}

bool hf_net::LoadHFNetTRModel() {
    if (LoadEngineFromFile(mStrEngineFile)) {
        return true;
    }

    spdlog::info("Start to build HFNet engine. It will take a while...");
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        return false;
    }
    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }

    auto constructed = construct_network(builder, config, parser);
    if (!constructed) {
        return false;
    }

    network->getInput(0)->setDimensions(mInputShape);

    auto profile_stream = samplesCommon::makeCudaStream();
    if (!profile_stream) {
        return false;
    }
    config->setProfileStream(*profile_stream);

    TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!mEngine) {
        return false;
    }

    mContext = std::shared_ptr<IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext) {
        return false;
    }

    SaveEngineToFile(mStrEngineFile, mEngine->serialize());

    return true;
}
bool hf_net::LoadEngineFromFile(const std::string& strEngineSaveFile) {
    std::ifstream engineFile(strEngineSaveFile, std::ios::binary);
    if (!engineFile.good()) {
        spdlog::error("Error opening engine file: {}. Will generate one!", strEngineSaveFile);
        return false;
    }
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> vecEngineBlob(fsize);
    engineFile.read(reinterpret_cast<char*>(vecEngineBlob.data()), fsize);
    if (!engineFile.good()) {
        std::cerr << "Error opening engine file: " << strEngineSaveFile << std::endl;
        return false;
    }

    std::unique_ptr<IRuntime> runtime{createInferRuntime(sample::gLogger)};
    if (!runtime) {
        return false;
    }

    mEngine.reset(runtime->deserializeCudaEngine(vecEngineBlob.data(), vecEngineBlob.size()));
    if (!mEngine) {
        return false;
    }

    mContext = std::shared_ptr<IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext) {
        return false;
    }
    spdlog::info("Deserialized engine file: {}!", strEngineSaveFile);
    return true;
}
bool hf_net::construct_network(TensorRTUniquePtr<IBuilder>& builder,
                               TensorRTUniquePtr<IBuilderConfig>& config,
                               TensorRTUniquePtr<nvonnxparser::IParser>& parser) const {
    auto parsed = parser->parseFromFile(mStrONNXFile.c_str(),
                                        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2 << 20);
    config->setFlag(BuilderFlag::kFP16);
    samplesCommon::enableDLA(builder.get(), config.get(), -1);
    return true;
}
bool hf_net::SaveEngineToFile(const std::string& strEngineSaveFile, nvinfer1::IHostMemory* serializedEngine) {
    std::ofstream engineFile(strEngineSaveFile, std::ios::binary);
    engineFile.write(reinterpret_cast<char const*>(serializedEngine->data()), (long)serializedEngine->size());
    if (engineFile.fail()) {
        spdlog::error("Saving engine to file failed.");
        return false;
    }
    return true;
}
bool hf_net::compute_global_descriptors(const cv::Mat& img, cv::Mat& globalDescriptors) {
    assert(mEngine);
    assert(not img.empty());
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(mInputShape.d[2], mInputShape.d[1]));
    // prepare input
    Mat2Tensor(resizedImg, input_tensors[0]);
    // infer
    if (!infer()) {
        return false;
    }
    // get output
    GetGlobalDescriptorFromTensor(output_tensors[2], globalDescriptors);

    return true;
}
void hf_net::Mat2Tensor(const cv::Mat& mat, RTTensor& tensor) {
    cv::Mat fromMat(mat.rows, mat.cols, CV_32FC(mat.channels()), static_cast<float*>(tensor.data));
    mat.convertTo(fromMat, CV_32F);
}
bool hf_net::infer() {
    if (input_tensors.empty()) {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    mpBuffers->copyInputToDevice();
    bool status = mContext->executeV2(mpBuffers->getDeviceBindings().data());
    if (!status) {
        spdlog::warn("Failed to execute hf");
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    mpBuffers->copyOutputToHost();

    return true;
}
void hf_net::GetGlobalDescriptorFromTensor(const RTTensor& tDescriptors, cv::Mat& globalDescriptors) {
    auto vResGlobalDescriptor = static_cast<float*>(tDescriptors.data);
    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp) {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor[temp];
    }
}

} // namespace stella_vslam::hloc