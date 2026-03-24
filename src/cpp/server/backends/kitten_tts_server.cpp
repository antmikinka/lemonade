#include "lemon/backends/kitten_tts_server.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backend_manager.h"
#include "lemon/utils/process_manager.h"
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
    params.repo = "second-state/kitten_tts_rs";

#ifdef _WIN32
    params.filename = "kitten-tts-x86_64-windows.zip";
#elif defined(__linux__)
    params.filename = "kitten-tts-x86_64-linux.tar.gz";
#else
    throw std::runtime_error("Unsupported platform for kitten-tts");
#endif

    return params;
}

KittenTtsServer::KittenTtsServer(const std::string& log_level, ModelManager* model_manager, BackendManager* backend_manager)
    : WrappedServer("kitten-tts-server", log_level, model_manager, backend_manager) {

}

KittenTtsServer::~KittenTtsServer() {
    unload();
}

void KittenTtsServer::load(const std::string& model_name, const ModelInfo& model_info, const RecipeOptions& options, bool do_not_upgrade) {
    LOG(INFO, "KittenTtsServer") << "Loading model: " << model_name << std::endl;

    // Install kitten-tts if needed
    backend_manager_->install_backend(SPEC.recipe, "cpu");

    // Use pre-resolved model path
    fs::path model_path = fs::path(model_info.resolved_path());
    if (model_path.empty() || !fs::exists(model_path)) {
        throw std::runtime_error("Model file not found for checkpoint: " + model_info.checkpoint());
    }

    // Get kitten-tts-server executable path
    std::string exe_path = BackendUtils::get_backend_binary_path(SPEC, "cpu");

    // Choose a port
    port_ = choose_port();
    if (port_ == 0) {
        throw std::runtime_error("Failed to find an available port");
    }

    LOG(INFO, "KittenTtsServer") << "Starting server on port " << port_ << std::endl;

    // Build command line arguments
    // kitten-tts-server takes model directory as first argument, then --host and --port
    std::vector<std::string> args = {
        model_path.string(),
        "--host", "127.0.0.1",
        "--port", std::to_string(port_)
    };

    // Set up environment variables for espeak-ng data path
    std::vector<std::pair<std::string, std::string>> env_vars;
    fs::path exe_dir = fs::path(exe_path).parent_path();

#ifndef _WIN32
    // On Linux, set LD_LIBRARY_PATH to include the binary directory
    std::string lib_path = exe_dir.string();
    const char* existing_ld_path = std::getenv("LD_LIBRARY_PATH");
    if (existing_ld_path && strlen(existing_ld_path) > 0) {
        lib_path = lib_path + ":" + std::string(existing_ld_path);
    }
    env_vars.push_back({"LD_LIBRARY_PATH", lib_path});
    LOG(INFO, "KittenTtsServer") << "Setting LD_LIBRARY_PATH=" << lib_path << std::endl;
#endif

    // Launch the subprocess
    process_handle_ = utils::ProcessManager::start_process(
        exe_path,
        args,
        "",     // working_dir (empty = current)
        is_debug(),  // inherit_output
        false,
        env_vars
    );

    if (process_handle_.pid == 0) {
        throw std::runtime_error("Failed to start kitten-tts-server process");
    }

    LOG(INFO, "KittenTtsServer") << "Process started with PID: " << process_handle_.pid << std::endl;

    // Wait for server to be ready
    if (!wait_for_ready("/health")) {
        unload();
        throw std::runtime_error("kitten-tts-server failed to start or become ready");
    }
}

void KittenTtsServer::unload() {
    if (process_handle_.pid != 0) {
        LOG(INFO, "KittenTtsServer") << "Stopping server (PID: " << process_handle_.pid << ")" << std::endl;
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

void KittenTtsServer::audio_speech(const json& request, httplib::DataSink& sink) {
    json tts_request = request;
    tts_request["model"] = "kitten-tts";

    // Forward the request to the kitten-tts-server
    // The server supports OpenAI-compatible /v1/audio/speech endpoint
    forward_streaming_request("/v1/audio/speech", tts_request.dump(), sink, false);
}

} // namespace backends
} // namespace lemon
