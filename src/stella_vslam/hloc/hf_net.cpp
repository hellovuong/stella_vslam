//
// Created by vuong on 3/14/23.
//

#include "hf_net.h"

namespace stella_vslam::hloc {

hf_net::hf_net(const hfnet_params& params) {
    mStrONNXFile = params.onnx_model_path;
    mStrCacheFile = params.cache_path;

    mInputShape = {1, params.image_height, params.image_width, 1};
    sample::setReportableSeverity(sample::Logger::Severity::kERROR);
    if (LoadHFNetTRModel())
        return;
}

bool hf_net::LoadHFNetTRModel() {
    if (LoadEngineFromFile(mStrEngineFile))
        return true;

    spdlog::info("Start to build HFNet engine. It will take a while...");
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(
                                    NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
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

    network->getInput(0)->setDimensions(mInputShape);

    auto constructed = construct_network(builder, config, parser);
    if (!constructed) {
        return false;
    }

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

    SaveEngineToFile(mStrEngineFile, )

    return true;
}
bool hf_net::LoadEngineFromFile(const std::string& strEngineSaveFile) {
    std::ifstream engineFile(strEngineSaveFile, std::ios::binary);
    if (!engineFile.good()) {
        std::cerr << "Error opening engine file: " << strEngineSaveFile << std::endl;
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
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512_MiB);
    config->setFlag(BuilderFlag::kFP16);
    samplesCommon::enableDLA(builder.get(), config.get(), false);
    return true;
}
bool hf_net::SaveEngineToFile(const std::string& strEngineSaveFile, const std::unique_ptr<nvinfer1::IHostMemory>& serializedEngine) {
    return false;
}

} // namespace stella_vslam::hloc