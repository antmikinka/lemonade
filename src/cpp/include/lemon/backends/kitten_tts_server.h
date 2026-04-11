// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 Lemonade Project

#pragma once

#include "lemon/wrapped_server.h"
#include "lemon/server_capabilities.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backends/kitten_tts_native.h"
#include "lemon/backends/audio_encoder.h"
#include <string>
#include <filesystem>
#include <memory>

namespace lemon {
namespace backends {

/**
 * KittenTTS Server - Native C++ Implementation
 *
 * This class provides native in-process TTS inference using ONNX Runtime,
 * replacing the subprocess-based approach. It supports:
 * - 8 voices (bella, jasper, luna, bruno, rosie, hugo, kiki, leo)
 * - OpenAI-compatible /v1/audio/speech endpoint
 * - Multiple output formats (mp3, opus, wav, pcm)
 * - Speed control (0.5x to 2.0x)
 * - 24kHz audio output
 */
class KittenTtsServer : public WrappedServer, public ITextToSpeechServer {
public:
#ifndef LEMONADE_TRAY
    static InstallParams get_install_params(const std::string& backend, const std::string& version);
#endif

    inline static const BackendSpec SPEC = BackendSpec(
            "kitten-tts",
    #ifdef _WIN32
            "kitten-tts-server.exe"
    #else
            "kitten-tts-server"
    #endif
#ifndef LEMONADE_TRAY
        , get_install_params
#endif
    );

    explicit KittenTtsServer(const std::string& log_level,
                          ModelManager* model_manager,
                          BackendManager* backend_manager);

    ~KittenTtsServer() override;

    void load(const std::string& model_name,
             const ModelInfo& model_info,
             const RecipeOptions& options,
             bool do_not_upgrade) override;

    void unload() override;

    // ICompletionServer implementation (not supported - return errors)
    json chat_completion(const json& request) override;
    json completion(const json& request) override;
    json responses(const json& request) override;

    // ITextToSpeechServer implementation
    void audio_speech(const json& request, httplib::DataSink& sink) override;

    /**
     * Get the native TTS engine instance.
     * @return Pointer to the native engine (nullptr if not loaded)
     */
    KittenTtsNative* get_native_engine() const { return tts_engine_.get(); }

    /**
     * Check if native engine is loaded.
     * @return true if native engine is loaded
     */
    bool is_native_loaded() const { return tts_engine_ && tts_engine_->is_loaded(); }

private:
    // Native TTS engine (replaces subprocess approach)
    std::unique_ptr<KittenTtsNative> tts_engine_;

    // Helper method to encode and stream audio
    void encode_and_stream(
        const std::vector<float>& audio,
        const std::string& format,
        httplib::DataSink& sink
    );
};

} // namespace backends
} // namespace lemon
