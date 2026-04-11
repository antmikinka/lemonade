#include "lemon/backends/kitten_tts_native.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace lemon {
namespace backends {

KittenTtsEngine::KittenTtsEngine() = default;

KittenTtsEngine::~KittenTtsEngine() = default;

bool KittenTtsEngine::initialize(const std::string& model_dir) {
    model_dir_ = model_dir;

    // Check for required files
    fs::path model_path = fs::path(model_dir) / "model.onnx";
    fs::path voices_path = fs::path(model_dir) / "voices.npz";

    if (!fs::exists(model_path)) {
        std::cerr << "Model file not found: " << model_path << std::endl;
        return false;
    }

    if (!fs::exists(voices_path)) {
        std::cerr << "Voices file not found: " << voices_path << std::endl;
        return false;
    }

    // Initialize components
    phonemizer_ = std::make_unique<KittenTtsPhonemizer>();
    if (!phonemizer_->initialize()) {
        std::cerr << "Failed to initialize phonemizer" << std::endl;
        return false;
    }

    voice_manager_ = std::make_unique<KittenTtsVoiceManager>();
    if (!voice_manager_->load_voices(voices_path.string())) {
        std::cerr << "Failed to load voices" << std::endl;
        return false;
    }

    onnx_model_ = std::make_unique<KittenTtsOnnxModel>();
    if (!onnx_model_->load_model(model_path.string())) {
        std::cerr << "Failed to load ONNX model" << std::endl;
        return false;
    }

    preprocessor_ = std::make_unique<KittenTtsPreprocessor>();

    initialized_ = true;
    std::cout << "KittenTTS engine initialized successfully" << std::endl;
    return true;
}

std::vector<float> KittenTtsEngine::synthesize(
    const std::string& text,
    const std::string& voice_name,
    float speed
) {
    if (!initialized_) {
        std::cerr << "Engine not initialized" << std::endl;
        return {};
    }

    // Get voice embedding
    const std::vector<float>* voice_embedding = voice_manager_->get_voice(voice_name);
    if (!voice_embedding) {
        std::cerr << "Voice not found: " << voice_name << std::endl;
        return {};
    }

    // Preprocess text
    std::string processed_text = preprocessor_->preprocess(text);

    // Phonemize
    std::string phonemes = phonemizer_->phonemize(processed_text);
    if (phonemes.empty()) {
        std::cerr << "Phonemization failed" << std::endl;
        return {};
    }

    // Tokenize
    std::vector<int> input_ids = preprocessor_->tokenize(phonemes);

    // Run inference
    std::vector<float> audio = onnx_model_->infer(input_ids, *voice_embedding, speed);

    return audio;
}

std::vector<std::string> KittenTtsEngine::get_available_voices() const {
    if (!initialized_) {
        return {};
    }
    return voice_manager_->get_available_voices();
}

int KittenTtsEngine::get_sample_rate() const {
    return voice_manager_->get_sample_rate();
}

} // namespace backends
} // namespace lemon
