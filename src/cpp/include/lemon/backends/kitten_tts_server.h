// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 Lemonade Project

#pragma once

#include "lemon/wrapped_server.h"
#include "lemon/server_capabilities.h"
#include "lemon/backends/backend_utils.h"
#include <string>
#include <filesystem>

namespace lemon {
namespace backends {

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
};

} // namespace backends
} // namespace lemon
