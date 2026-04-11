#include "lemon/backends/kitten_tts_onnx_model.h"
#include <iostream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

namespace lemon {
namespace backends {

KittenTtsOnnxModel::KittenTtsOnnxModel() = default;

KittenTtsOnnxModel::~KittenTtsOnnxModel() {
    if (session_) {
        delete session_;
        session_ = nullptr;
    }
}

bool KittenTtsOnnxModel::load_model(const std::string& model_path) {
    try {
#ifdef _WIN32
        // Convert UTF-8 path to UTF-16 for Windows
        int wide_len = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
        std::wstring wide_path(wide_len, 0);
        MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &wide_path[0], wide_len);

        session_ = new Ort::Session(env_, wide_path.c_str(), session_options_);
#else
        session_ = new Ort::Session(env_, model_path.c_str(), session_options_);
#endif

        // Get input/output info
        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            input_names_.push_back(name.get());

            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shapes_.push_back(tensor_info.GetShape());
        }

        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            auto name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            output_names_.push_back(name.get());

            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shapes_.push_back(tensor_info.GetShape());
        }

        std::cout << "Loaded ONNX model with " << num_inputs << " inputs and " << num_outputs << " outputs" << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load ONNX model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> KittenTtsOnnxModel::infer(
    const std::vector<int>& input_ids,
    const std::vector<float>& style_vector,
    float speed
) {
    if (!session_) {
        std::cerr << "Model not loaded" << std::endl;
        return {};
    }

    try {
        // Create input tensors
        std::vector<Ort::Value> inputs;

        // Input 1: input_ids (sequence of token IDs)
        std::vector<int64_t> input_ids_64(input_ids.begin(), input_ids.end());
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
            input_ids_64.data(),
            input_ids_64.size(),
            input_shape.data(),
            input_shape.size()
        ));

        // Input 2: style_vector (voice embedding)
        std::vector<int64_t> style_shape = {1, static_cast<int64_t>(style_vector.size())};
        inputs.push_back(Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
            const_cast<float*>(style_vector.data()),
            style_vector.size(),
            style_shape.data(),
            style_shape.size()
        ));

        // Input 3: speed (scalar)
        std::vector<int64_t> speed_shape = {1};
        inputs.push_back(Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
            &speed,
            1,
            speed_shape.data(),
            speed_shape.size()
        ));

        // Run inference
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            inputs.data(),
            inputs.size(),
            output_names_.data(),
            output_names_.size()
        );

        // Extract output (should be audio waveform)
        if (outputs.empty()) {
            std::cerr << "No output from model" << std::endl;
            return {};
        }

        float* output_data = outputs[0].GetTensorMutableData<float>();
        size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> result(output_data, output_data + output_size);
        return result;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX inference failed: " << e.what() << std::endl;
        return {};
    }
}

} // namespace backends
} // namespace lemon
