#pragma once

#include "kitten_tts_phonemizer.h"
#include "kitten_tts_voice.h"
#include "kitten_tts_onnx_model.h"
#include "kitten_tts_preprocessor.h"
#include <string>
#include <vector>
#include <memory>

namespace lemon {
namespace backends {

/**
 * Main KittenTTS engine orchestrating all components.
 * Synthesizes speech from text using ONNX inference.
 */
class KittenTtsNative {
public:
    KittenTtsNative();
    ~KittenTtsNative();

    // Initialize engine with model directory
    bool initialize(const std::string& model_dir);

    // Synthesize speech from text
    std::vector<float> synthesize(
        const std::string& text,
        const std::string& voice_name,
        float speed = 1.0f
    );

    // Get available voices
    std::vector<std::string> get_available_voices() const;

    // Get sample rate
    int get_sample_rate() const;

    // Check if loaded
    bool is_loaded() const { return initialized_; }

private:
    bool initialized_ = false;
    std::string model_dir_;

    std::unique_ptr<KittenTtsPhonemizer> phonemizer_;
    std::unique_ptr<KittenTtsVoiceManager> voice_manager_;
    std::unique_ptr<KittenTtsOnnxModel> onnx_model_;
    std::unique_ptr<KittenTtsPreprocessor> preprocessor_;
};

} // namespace backends
} // namespace lemon
