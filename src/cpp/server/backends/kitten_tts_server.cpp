// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 Lemonade Project

#include "lemon/backends/kitten_tts_server.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backends/kitten_tts_native.h"
#include "lemon/backends/audio_encoder.h"
#include "lemon/backend_manager.h"
#include "lemon/utils/json_utils.h"
#include "lemon/error_types.h"
#include <httplib.h>
#include <iostream>
#include <vector>
#include <lemon/utils/aixlog.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace lemon::utils;

namespace lemon {
namespace backends {

InstallParams KittenTtsServer::get_install_params(const std::string& /*backend*/, const std::string& version) {
    InstallParams params;
    // For native implementation, we don't need external binaries
    // The model files are downloaded via ModelManager
    params.repo = "KittenML/KittenTTS";

#ifdef _WIN32
    params.filename = "kitten-tts-models.zip";
#elif defined(__linux__)
    params.filename = "kitten-tts-models.tar.gz";
#else
    params.filename = "kitten-tts-models.tar.gz";
#endif

    return params;
}

KittenTtsServer::KittenTtsServer(const std::string& log_level, ModelManager* model_manager, BackendManager* backend_manager)
    : WrappedServer("kitten-tts-native", log_level, model_manager, backend_manager) {
    LOG(INFO, "KittenTtsServer") << "KittenTTS Native Server initialized" << std::endl;
}

KittenTtsServer::~KittenTtsServer() {
    unload();
}

void KittenTtsServer::load(const std::string& model_name, const ModelInfo& model_info, const RecipeOptions& options, bool do_not_upgrade) {
    LOG(INFO, "KittenTtsServer") << "Loading KittenTTS model (native): " << model_name << std::endl;

    // Use pre-resolved model path from ModelManager
    fs::path model_path = fs::path(model_info.resolved_path());
    if (model_path.empty() || !fs::exists(model_path)) {
        throw std::runtime_error("Model file not found for checkpoint: " + model_info.checkpoint());
    }

    LOG(INFO, "KittenTtsServer") << "Model path: " << model_path.string() << std::endl;

    // Verify model files exist
    fs::path model_onnx = model_path / "model.onnx";
    fs::path voices_npz = model_path / "voices.npz";

    if (!fs::exists(model_onnx)) {
        throw std::runtime_error("Model file not found: " + model_onnx.string());
    }

    if (!fs::exists(voices_npz)) {
        throw std::runtime_error("Voice file not found: " + voices_npz.string());
    }

    // Create and initialize the native TTS engine
    tts_engine_ = std::make_unique<KittenTtsNative>(model_path.string(), log_level_);

    // Set espeak-ng data path if available
    // Check common locations
    std::string espeak_data_path;

#ifdef _WIN32
    // Windows: check relative to executable
    fs::path exe_path = fs::weakly_canonical(fs::current_path());
    if (fs::exists(exe_path / "espeak-ng-data")) {
        espeak_data_path = (exe_path / "espeak-ng-data").string();
    }
#elif defined(__APPLE__)
    // macOS: check bundle resources
    fs::path resources_path = exe_path / "../Resources/espeak-ng-data";
    if (fs::exists(resources_path)) {
        espeak_data_path = resources_path.string();
    }
#else
    // Linux: check system path
    if (fs::exists("/usr/share/espeak-ng-data")) {
        espeak_data_path = "/usr/share/espeak-ng-data";
    }
#endif

    if (!espeak_data_path.empty()) {
        tts_engine_->set_espeak_data_path(espeak_data_path);
        LOG(INFO, "KittenTtsServer") << "Using espeak-ng data from: " << espeak_data_path << std::endl;
    } else {
        LOG(WARNING, "KittenTtsServer") << "espeak-ng data path not set, using system default" << std::endl;
    }

    // Load the engine (model + voices)
    try {
        tts_engine_->load();
        LOG(INFO, "KittenTtsServer") << "KittenTTS native engine loaded successfully" << std::endl;

        // Log available voices
        auto voices = tts_engine_->get_voices();
        LOG(INFO, "KittenTtsServer") << "Available voices: ";
        for (const auto& voice : voices) {
            LOG(INFO, "KittenTtsServer") << voice << " ";
        }
        LOG(INFO, "KittenTtsServer") << std::endl;

    } catch (const std::exception& e) {
        LOG(ERROR, "KittenTtsServer") << "Failed to load KittenTTS engine: " << e.what() << std::endl;
        tts_engine_.reset();
        throw std::runtime_error("Failed to initialize KittenTTS native engine: " + std::string(e.what()));
    }

    // Set model metadata
    set_model_metadata(model_name, model_info.checkpoint(),
                       model_info.type(), model_info.device(), options);

    LOG(INFO, "KittenTtsServer") << "KittenTTS native model ready" << std::endl;
}

void KittenTtsServer::unload() {
    // Native engine doesn't need explicit cleanup - destructor handles it
    if (tts_engine_) {
        LOG(INFO, "KittenTtsServer") << "Unloading KittenTTS native engine" << std::endl;
        tts_engine_.reset();
    }

    // Clean up any subprocess state (for compatibility)
    if (process_handle_.pid != 0) {
        utils::ProcessManager::stop_process(process_handle_);
        port_ = 0;
        process_handle_ = {nullptr, 0};
    }
}

// ICompletionServer implementation (not supported - return errors)
json KittenTtsServer::chat_completion(const json& request) {
    return json{
        {"error", {
            {"message", "Kitten-TTS does not support text completion. Use audio speech endpoints instead."},
            {"type", "unsupported_operation"},
            {"code", "model_not_applicable"}
        }}
    };
}

json KittenTtsServer::completion(const json& request) {
    return json{
        {"error", {
            {"message", "Kitten-TTS does not support text completion. Use audio speech endpoints instead."},
            {"type", "unsupported_operation"},
            {"code", "model_not_applicable"}
        }}
    };
}

json KittenTtsServer::responses(const json& request) {
    return json{
        {"error", {
            {"message", "Kitten-TTS does not support text completion. Use audio speech endpoints instead."},
            {"type", "unsupported_operation"},
            {"code", "model_not_applicable"}
        }}
    };
}

void KittenTtsServer::encode_and_stream(
    const std::vector<float>& audio,
    const std::string& format,
    httplib::DataSink& sink
) {
    int sample_rate = tts_engine_->get_sample_rate();

    if (format == "wav") {
        // WAV encoding - build complete file
        std::vector<uint8_t> wav_data;
        AudioEncoder::encode_wav(audio, sample_rate, wav_data);
        sink.write(reinterpret_cast<const char*>(wav_data.data()), wav_data.size());
    } else if (format == "mp3") {
        // MP3 encoding (streaming)
        if (!AudioEncoder::encode_mp3_streaming(audio, sample_rate, sink, 128)) {
            LOG(WARNING, "KittenTtsServer") << "MP3 encoding failed, falling back to WAV" << std::endl;
            std::vector<uint8_t> wav_data;
            AudioEncoder::encode_wav(audio, sample_rate, wav_data);
            sink.write(reinterpret_cast<const char*>(wav_data.data()), wav_data.size());
        }
    } else if (format == "opus") {
        // Opus encoding (streaming) - fallback to WAV
        if (!AudioEncoder::encode_opus_streaming(audio, sample_rate, sink, 96)) {
            LOG(WARNING, "KittenTtsServer") << "Opus encoding not available, falling back to WAV" << std::endl;
            std::vector<uint8_t> wav_data;
            AudioEncoder::encode_wav(audio, sample_rate, wav_data);
            sink.write(reinterpret_cast<const char*>(wav_data.data()), wav_data.size());
        }
    } else if (format == "aac") {
        // AAC encoding (streaming) - fallback to WAV
        if (!AudioEncoder::encode_aac_streaming(audio, sample_rate, sink, 128)) {
            LOG(WARNING, "KittenTtsServer") << "AAC encoding not available, falling back to WAV" << std::endl;
            std::vector<uint8_t> wav_data;
            AudioEncoder::encode_wav(audio, sample_rate, wav_data);
            sink.write(reinterpret_cast<const char*>(wav_data.data()), wav_data.size());
        }
    } else if (format == "pcm") {
        // Raw PCM (int16)
        AudioEncoder::write_pcm_streaming(audio, sink, true);
    } else if (format == "float") {
        // Raw PCM (float32)
        AudioEncoder::write_pcm_streaming(audio, sink, false);
    } else {
        // Default to WAV with warning
        LOG(WARNING, "KittenTtsServer") << "Unknown format: " << format << ", defaulting to WAV" << std::endl;
        std::vector<uint8_t> wav_data;
        AudioEncoder::encode_wav(audio, sample_rate, wav_data);
        sink.write(reinterpret_cast<const char*>(wav_data.data()), wav_data.size());
    }
}

void KittenTtsServer::audio_speech(const json& request, httplib::DataSink& sink) {
    if (!tts_engine_ || !tts_engine_->is_loaded()) {
        LOG(ERROR, "KittenTtsServer") << "TTS engine not loaded" << std::endl;
        // Send error response
        json error_response = json{
            {"error", {
                {"message", "TTS engine not loaded. Please wait for model to load."},
                {"type", "server_error"},
                {"code", "engine_not_loaded"}
            }}
        };
        sink.write(error_response.dump().c_str(), error_response.dump().size());
        sink.done = true;
        return;
    }

    // Extract request parameters
    std::string input_text = request.value("input", "");
    std::string voice = request.value("voice", "bella");  // Default voice
    std::string response_format = request.value("response_format", "mp3");
    float speed = request.value("speed", 1.0f);

    // Validate input
    if (input_text.empty()) {
        LOG(ERROR, "KittenTtsServer") << "Empty input text" << std::endl;
        json error_response = json{
            {"error", {
                {"message", "Empty input text provided"},
                {"type", "invalid_request"},
                {"code", "empty_input"}
            }}
        };
        sink.write(error_response.dump().c_str(), error_response.dump().size());
        sink.done = true;
        return;
    }

    // Check if voice is available
    if (!tts_engine_->has_voice(voice)) {
        auto available = tts_engine_->get_voices();
        LOG(WARNING, "KittenTtsServer") << "Voice '" << voice << "' not available, using default" << std::endl;
        voice = "bella";
    }

    // Validate response format
    if (!AudioEncoder::is_format_supported(response_format)) {
        LOG(ERROR, "KittenTtsServer") << "Unsupported format: " << response_format << std::endl;
        json error_response = json{
            {"error", {
                {"message", "Unsupported response format: " + response_format},
                {"type", "invalid_request"},
                {"code", "unsupported_format"}
            }}
        };
        sink.write(error_response.dump().c_str(), error_response.dump().size());
        sink.done = true;
        return;
    }

    // Clamp speed
    if (speed < 0.5f) speed = 0.5f;
    if (speed > 2.0f) speed = 2.0f;

    LOG(INFO, "KittenTtsServer") << "Synthesizing speech: '" << input_text.substr(0, 50)
                                  << (input_text.size() > 50 ? "..." : "")
                                  << "' voice=" << voice
                                  << " format=" << response_format
                                  << " speed=" << speed << std::endl;

    try {
        // Synthesize audio
        std::vector<float> audio;

        // Handle long texts with automatic chunking
        const size_t CHUNK_SIZE = 400;
        if (input_text.size() > CHUNK_SIZE) {
            audio = tts_engine_->synthesize_long(input_text, voice, speed, CHUNK_SIZE);
        } else {
            audio = tts_engine_->synthesize(input_text, voice, speed);
        }

        if (audio.empty()) {
            LOG(ERROR, "KittenTtsServer") << "Synthesis produced empty audio" << std::endl;
            json error_response = json{
                {"error", {
                    {"message", "Speech synthesis produced empty audio"},
                    {"type", "processing_error"},
                    {"code", "empty_output"}
                }}
            };
            sink.write(error_response.dump().c_str(), error_response.dump().size());
            sink.done = true;
            return;
        }

        // Set response content type
        std::string mime_type = AudioEncoder::get_mime_type(response_format);

        // Encode and stream the audio
        encode_and_stream(audio, response_format, sink);

        LOG(DEBUG, "KittenTtsServer") << "Speech synthesis complete: " << audio.size() << " samples" << std::endl;

    } catch (const std::exception& e) {
        LOG(ERROR, "KittenTtsServer") << "Synthesis error: " << e.what() << std::endl;
        json error_response = json{
            {"error", {
                {"message", std::string("Speech synthesis failed: ") + e.what()},
                {"type", "processing_error"},
                {"code", "synthesis_failed"}
            }}
        };
        sink.write(error_response.dump().c_str(), error_response.dump().size());
        sink.done = true;
    }
}

} // namespace backends
} // namespace lemon
