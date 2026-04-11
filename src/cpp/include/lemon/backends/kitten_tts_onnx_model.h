#pragma once

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace lemon {
namespace backends {

/**
 * ONNX Runtime wrapper for KittenTTS model inference.
 */
class KittenTtsOnnxModel {
public:
    KittenTtsOnnxModel();
    ~KittenTtsOnnxModel();

    // Load model from ONNX file
    bool load_model(const std::string& model_path);

    // Run inference
    std::vector<float> infer(
        const std::vector<int>& input_ids,
        const std::vector<float>& style_vector,
        float speed = 1.0f
    );

    // Check if loaded
    bool is_loaded() const { return session_ != nullptr; }

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "KittenTTS"};
    Ort::Session* session_ = nullptr;
    Ort::SessionOptions session_options_;

    // Input/output info
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
};

} // namespace backends
} // namespace lemon
