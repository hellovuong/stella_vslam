//
// Created by vuong on 4/6/23.
//

#include "stella_vslam/match/super_glue.h"
#include "spdlog/spdlog.h"

namespace stella_vslam::match {

SuperGlue::SuperGlue(SuperGlueConfig superglue_config)
    : superglue_config_(std::move(superglue_config)), engine_(nullptr) {
    sample::setReportableSeverity(sample::Logger::Severity::kVERBOSE);
    build();
}

bool SuperGlue::build() {
    if (deserialize_engine()) {
        return true;
    }
    spdlog::info("Start to build SuperGlue engine ...");
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }

    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicit_batch));
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

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        return false;
    }
    profile->setDimensions(superglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kMIN, Dims3(1, 1, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kOPT, Dims3(1, 1024, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kMAX, Dims3(1, 2048, 2));

    profile->setDimensions(superglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kMIN, Dims2(1, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kOPT, Dims2(1, 1024));
    profile->setDimensions(superglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kMAX, Dims2(1, 2048));

    profile->setDimensions(superglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kMIN, Dims3(1, 256, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kOPT, Dims3(1, 256, 1024));
    profile->setDimensions(superglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kMAX, Dims3(1, 256, 2048));

    profile->setDimensions(superglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kMIN, Dims3(1, 1, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kOPT, Dims3(1, 1024, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kMAX, Dims3(1, 2048, 2));

    profile->setDimensions(superglue_config_.input_tensor_names[4].c_str(), OptProfileSelector::kMIN, Dims2(1, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[4].c_str(), OptProfileSelector::kOPT, Dims2(1, 1024));
    profile->setDimensions(superglue_config_.input_tensor_names[4].c_str(), OptProfileSelector::kMAX, Dims2(1, 2048));

    profile->setDimensions(superglue_config_.input_tensor_names[5].c_str(), OptProfileSelector::kMIN, Dims3(1, 256, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[5].c_str(), OptProfileSelector::kOPT, Dims3(1, 256, 1024));
    profile->setDimensions(superglue_config_.input_tensor_names[5].c_str(), OptProfileSelector::kMAX, Dims3(1, 256, 2048));

    config->addOptimizationProfile(profile);

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

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        return false;
    }

    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
        }
    }

    save_engine();

    ASSERT(network->getNbInputs() == 6);
    spdlog::info("Built SuperGlue engine!");

    return true;
}
void SuperGlue::save_engine() {
    if (superglue_config_.engine_file.empty())
        return;
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory* data = engine_->serialize();
        std::ofstream file(superglue_config_.engine_file, std::ios::binary);
        if (!file)
            return;
        file.write(reinterpret_cast<const char*>(data->data()), (int)data->size());
    }
}
bool SuperGlue::deserialize_engine() {
    std::ifstream file(superglue_config_.engine_file, std::ios::binary);
    if (!file.good()) {
        spdlog::error("Error opening engine file: {}. Will generate one!", superglue_config_.engine_file);
        return false;
    }
    if (file.is_open()) {
        file.seekg(0, std::ifstream::end);
        int64_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);

        std::vector<uint8_t> vecEngineBlob(size);
        file.read(reinterpret_cast<char*>(vecEngineBlob.data()), size);
        file.close();

        IRuntime* runtime = createInferRuntime(sample::gLogger);
        if (not runtime) {
            return false;
        }

        engine_.reset(runtime->deserializeCudaEngine(vecEngineBlob.data(), vecEngineBlob.size()));
        if (!engine_) {
            spdlog::error("Fail to deserialize CudaEngine!");
            return false;
        }

        context_ = std::shared_ptr<IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            spdlog::error("Fail to create IExecutionContext!");
            return false;
        }

        spdlog::info("Deserialized SuperGlue engine: {}", superglue_config_.engine_file);
        return true;
    }
    return false;
}

bool SuperGlue::infer(const std::vector<cv::KeyPoint>& kps0, const cv::Mat& desc0, const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1,
                      Eigen::VectorXi& indices0, Eigen::VectorXi& indices1, Eigen::VectorXd& mscores0, Eigen::VectorXd& mscores1) {
    assert(engine_->getNbBindings() == 7);

    const int output_score_index = engine_->getBindingIndex(superglue_config_.output_tensor_names[0].c_str());

    context_->setBindingDimensions(0, Dims3(1, (int)kps0.size(), 2));
    context_->setBindingDimensions(1, Dims2(1, (int)kps0.size()));
    context_->setBindingDimensions(2, Dims3(1, 256, (int)kps0.size()));
    context_->setBindingDimensions(3, Dims3(1, (int)kps1.size(), 2));
    context_->setBindingDimensions(4, Dims2(1, (int)kps1.size()));
    context_->setBindingDimensions(5, Dims3(1, 256, (int)kps1.size()));

    output_scores_dims_ = context_->getBindingDimensions(output_score_index);

    buffers_manager_.reset();
    buffers_manager_ = std::make_unique<samplesCommon::BufferManager>(engine_, 0, context_.get());

    process_input(buffers_manager_, kps0, desc0, kps1, desc1);

    buffers_manager_->copyInputToDevice();

    bool status = context_->executeV2(buffers_manager_->getDeviceBindings().data());
    if (!status) {
        return false;
    }

    buffers_manager_->copyOutputToHost();

    // Verify results
    process_output(buffers_manager_, indices0, indices1, mscores0, mscores1);

    return true;
}
bool SuperGlue::construct_network(TensorRTUniquePtr<IBuilder>& builder, TensorRTUniquePtr<IBuilderConfig>& config,
                                  TensorRTUniquePtr<nvonnxparser::IParser>& parser) const {
    auto parsed = parser->parseFromFile(superglue_config_.onnx_file.c_str(),
                                        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512_MiB);
    config->setFlag(BuilderFlag::kFP16);
    samplesCommon::enableDLA(builder.get(), config.get(), superglue_config_.dla_core);
    return true;
}
bool SuperGlue::process_input(const std::unique_ptr<samplesCommon::BufferManager>& buffers,
                              const std::vector<cv::KeyPoint>& kps0, const cv::Mat& desc0,
                              const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1) {
    auto* keypoints_0_buffer = static_cast<float*>(buffers->getHostBuffer(superglue_config_.input_tensor_names[0]));
    auto* scores_0_buffer = static_cast<float*>(buffers->getHostBuffer(superglue_config_.input_tensor_names[1]));
    auto* descriptors_0_buffer = static_cast<float*>(buffers->getHostBuffer(superglue_config_.input_tensor_names[2]));

    auto* keypoints_1_buffer = static_cast<float*>(buffers->getHostBuffer(superglue_config_.input_tensor_names[3]));
    auto* scores_1_buffer = static_cast<float*>(buffers->getHostBuffer(superglue_config_.input_tensor_names[4]));
    auto* descriptors_1_buffer = static_cast<float*>(buffers->getHostBuffer(superglue_config_.input_tensor_names[5]));

    // score
    for (int cols0 = 0; cols0 < (int)kps0.size(); ++cols0) {
        scores_0_buffer[cols0] = kps0[cols0].response;
    }
    // points
    for (int colk0 = 0; colk0 < (int)kps0.size(); ++colk0) {
        keypoints_0_buffer[colk0 * 2] = kps0[colk0].pt.x;
        keypoints_0_buffer[colk0 * 2 + 1] = kps0[colk0].pt.y;
    }
    // desc
    for (int cold0 = 0; cold0 < desc0.cols; ++cold0) {
        for (int rowd0 = 0; rowd0 < desc0.rows; ++rowd0) {
            descriptors_0_buffer[cold0 * desc0.rows + rowd0] = desc0.at<float>(rowd0, cold0);
        }
    }

    // score
    for (int cols0 = 0; cols0 < (int)kps1.size(); ++cols0) {
        scores_1_buffer[cols0] = kps1[cols0].response;
    }
    // points
    for (int colk0 = 0; colk0 < (int)kps1.size(); ++colk0) {
        keypoints_1_buffer[colk0 * 2] = kps1[colk0].pt.x;
        keypoints_1_buffer[colk0 * 2 + 1] = kps1[colk0].pt.y;
    }
    // desc
    for (int cold0 = 0; cold0 < desc1.cols; ++cold0) {
        for (int rowd0 = 0; rowd0 < desc1.rows; ++rowd0) {
            descriptors_1_buffer[cold0 * desc1.rows + rowd0] = desc1.at<float>(rowd0, cold0);
        }
    }
    return true;
}
void where_negative_one(const int* flag_data, const int* data, int size, std::vector<int>& indices) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            indices.push_back(data[i]);
        }
        else {
            indices.push_back(-1);
        }
    }
}

void max_matrix(const float* data, int* indices, float* values, int h, int w, int dim) {
    if (dim == 2) {
        for (int i = 0; i < h - 1; ++i) {
            float max_value = -FLT_MAX;
            int max_indices = 0;
            for (int j = 0; j < w - 1; ++j) {
                if (max_value < data[i * w + j]) {
                    max_value = data[i * w + j];
                    max_indices = j;
                }
            }
            values[i] = max_value;
            indices[i] = max_indices;
        }
    }
    else if (dim == 1) {
        for (int i = 0; i < w - 1; ++i) {
            float max_value = -FLT_MAX;
            int max_indices = 0;
            for (int j = 0; j < h - 1; ++j) {
                if (max_value < data[j * w + i]) {
                    max_value = data[j * w + i];
                    max_indices = j;
                }
            }
            values[i] = max_value;
            indices[i] = max_indices;
        }
    }
}

void equal_gather(const int* indices0, const int* indices1, int* mutual, int size) {
    for (int i = 0; i < size; ++i) {
        if (indices0[indices1[i]] == i) {
            mutual[i] = 1;
        }
        else {
            mutual[i] = 0;
        }
    }
}

void where_exp(const int* flag_data, float* data, std::vector<double>& mscores0, int size) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            mscores0.push_back(std::exp(data[i]));
        }
        else {
            mscores0.push_back(0);
        }
    }
}

void where_gather(const int* flag_data, int* indices, std::vector<double>& mscores0, std::vector<double>& mscores1, int size) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            mscores1.push_back(mscores0[indices[i]]);
        }
        else {
            mscores1.push_back(0);
        }
    }
}

void and_threshold(const int* mutual0, int* valid0, const std::vector<double>& mscores0, double threhold) {
    for (int i = 0; i < (int)mscores0.size(); ++i) {
        if (mutual0[i] == 1 && mscores0[i] > threhold) {
            valid0[i] = 1;
        }
        else {
            valid0[i] = 0;
        }
    }
}

void and_gather(const int* mutual1, const int* valid0, const int* indices1, int* valid1, int size) {
    for (int i = 0; i < size; ++i) {
        if (mutual1[i] == 1 && valid0[indices1[i]] == 1) {
            valid1[i] = 1;
        }
        else {
            valid1[i] = 0;
        }
    }
}

void decode(float* scores, int h, int w, std::vector<int>& indices0, std::vector<int>& indices1, std::vector<double>& mscores0, std::vector<double>& mscores1) {
    auto* max_indices0 = new int[h - 1];
    auto* max_indices1 = new int[w - 1];
    auto* max_values0 = new float[h - 1];
    auto* max_values1 = new float[w - 1];

    max_matrix(scores, max_indices0, max_values0, h, w, 2);
    max_matrix(scores, max_indices1, max_values1, h, w, 1);

    auto* mutual0 = new int[h - 1];
    auto* mutual1 = new int[w - 1];

    equal_gather(max_indices1, max_indices0, mutual0, h - 1);
    equal_gather(max_indices0, max_indices1, mutual1, w - 1);
    where_exp(mutual0, max_values0, mscores0, h - 1);
    where_gather(mutual1, max_indices1, mscores0, mscores1, w - 1);

    auto* valid0 = new int[h - 1];
    auto* valid1 = new int[w - 1];

    and_threshold(mutual0, valid0, mscores0, 0.2);
    and_gather(mutual1, valid0, max_indices1, valid1, w - 1);
    where_negative_one(valid0, max_indices0, h - 1, indices0);
    where_negative_one(valid1, max_indices1, w - 1, indices1);

    delete[] max_indices0;
    delete[] max_indices1;
    delete[] max_values0;
    delete[] max_values1;
    delete[] mutual0;
    delete[] mutual1;
    delete[] valid0;
    delete[] valid1;
}

bool SuperGlue::process_output(const std::unique_ptr<samplesCommon::BufferManager>& buffers, Eigen::VectorXi& indices0, Eigen::VectorXi& indices1,
                               Eigen::VectorXd& mscores0, Eigen::VectorXd& mscores1) {
    indices0_.clear();
    indices1_.clear();
    mscores0_.clear();
    mscores1_.clear();
    auto* output_score = static_cast<float*>(buffers->getHostBuffer(superglue_config_.output_tensor_names[0]));

    int scores_map_h = output_scores_dims_.d[1];
    int scores_map_w = output_scores_dims_.d[2];

    auto* scores = new float[(scores_map_h + 1) * (scores_map_w + 1)];

    decode(output_score, scores_map_h, scores_map_w, indices0_, indices1_, mscores0_, mscores1_);

    indices0.resize((int)indices0_.size());
    indices1.resize((int)indices1_.size());
    mscores0.resize((int)mscores0_.size());
    mscores1.resize((int)mscores1_.size());

    for (int i0 = 0; i0 < (int)indices0_.size(); ++i0) {
        indices0(i0) = indices0_[i0];
    }
    for (int i1 = 0; i1 < (int)indices1_.size(); ++i1) {
        indices1(i1) = indices1_[i1];
    }
    for (int j0 = 0; j0 < (int)mscores0_.size(); ++j0) {
        mscores0(j0) = mscores0_[j0];
    }
    for (int j1 = 0; j1 < (int)mscores1_.size(); ++j1) {
        mscores1(j1) = mscores1_[j1];
    }
    return true;
}

int SuperGlue::MatchingPoints(const std::vector<cv::KeyPoint>& kps0, const cv::Mat& desc0,
                              const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1,
                              std::vector<cv::DMatch>& matches) {
    matches.clear();

    auto norm_features0 = NormalizeKeypoints(kps0, superglue_config_.image_width, superglue_config_.image_height);
    auto norm_features1 = NormalizeKeypoints(kps1, superglue_config_.image_width, superglue_config_.image_height);

    Eigen::VectorXi indices0, indices1;
    Eigen::VectorXd mscores0, mscores1;

    infer(norm_features0, desc0, norm_features1, desc1, indices0, indices1, mscores0, mscores1);

    for (int i = 0; i < (int)indices0.size(); i++) {
        if (indices0(i) < indices1.size() && indices0(i) >= 0 && indices1(indices0(i)) == i) {
            double d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
            // i - queryIdx, indices0 - trainIdx
            matches.emplace_back(i, indices0[i], d);
        }
    }

    return (int)matches.size();
}

void SuperGlue::gen_configs(const YAML::Node& superglue_node, SuperGlueConfig& superglue_config) {
    superglue_config.image_width = superglue_node["image_width"].as<int>();
    superglue_config.image_height = superglue_node["image_height"].as<int>();
    superglue_config.dla_core = superglue_node["dla_core"].as<int>();
    YAML::Node superglue_input_tensor_names_node = superglue_node["input_tensor_names"];
    size_t superglue_num_input_tensor_names = superglue_input_tensor_names_node.size();

    for (size_t i = 0; i < superglue_num_input_tensor_names; i++) {
        superglue_config.input_tensor_names.push_back(superglue_input_tensor_names_node[i].as<std::string>());
    }

    YAML::Node superglue_output_tensor_names_node = superglue_node["output_tensor_names"];
    size_t superglue_num_output_tensor_names = superglue_output_tensor_names_node.size();
    for (size_t i = 0; i < superglue_num_output_tensor_names; i++) {
        superglue_config.output_tensor_names.push_back(superglue_output_tensor_names_node[i].as<std::string>());
    }

    auto superglue_onnx_file = superglue_node["onnx_file"].as<std::string>();
    auto superglue_engine_file = superglue_node["engine_file"].as<std::string>();
    superglue_config.onnx_file = superglue_onnx_file;
    superglue_config.engine_file = superglue_engine_file;
}

} // namespace stella_vslam::match