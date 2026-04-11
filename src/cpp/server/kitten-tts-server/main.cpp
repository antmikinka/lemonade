/**
 * KittenTTS Standalone Server
 *
 * A lightweight HTTP server that provides OpenAI-compatible TTS API
 * using the KittenTTS model with ONNX Runtime inference.
 */

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include "lemon/backends/kitten_tts_native.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

// Global engine instance
std::unique_ptr<lemon::backends::KittenTtsNative> g_engine;
int g_port = 8780;
std::string g_host = "127.0.0.1";

// WAV header structure
struct WavHeader {
    uint8_t riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size;
    uint8_t wave[4] = {'W', 'A', 'V', 'E'};
    uint8_t fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;  // PCM
    uint16_t num_channels = 1;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align = 2;  // 16-bit mono
    uint16_t bits_per_sample = 16;
    uint8_t data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;

    WavHeader(uint32_t data_sz, uint32_t rate)
        : file_size(data_sz + 36),
          sample_rate(rate),
          byte_rate(rate * 2),
          data_size(data_sz) {}
};

// Encode audio to WAV format
std::vector<uint8_t> encode_wav(const std::vector<float>& audio, int sample_rate) {
    // Convert float32 to int16
    std::vector<int16_t> int16_audio(audio.size());
    for (size_t i = 0; i < audio.size(); i++) {
        float val = std::max(-1.0f, std::min(1.0f, audio[i]));
        int16_audio[i] = static_cast<int16_t>(val * 32767);
    }

    // Create WAV header
    uint32_t data_size = static_cast<uint32_t>(int16_audio.size() * sizeof(int16_t));
    WavHeader header(data_size, sample_rate);

    // Combine header and data
    std::vector<uint8_t> result(sizeof(WavHeader) + data_size);
    std::memcpy(result.data(), &header, sizeof(WavHeader));
    std::memcpy(result.data() + sizeof(WavHeader), int16_audio.data(), data_size);

    return result;
}

// Health check endpoint
void handle_health(const httplib::Request& req, httplib::Response& res) {
    if (g_engine && g_engine->is_loaded()) {
        res.set_content(R"({"status":"healthy"})", "application/json");
    } else {
        res.status = 503;
        res.set_content(R"({"status":"unhealthy"})", "application/json");
    }
}

// Get available voices
void handle_voices(const httplib::Request& req, httplib::Response& res) {
    if (!g_engine || !g_engine->is_loaded()) {
        res.status = 503;
        res.set_content(R"({"error":"Engine not initialized"})", "application/json");
        return;
    }

    auto voices = g_engine->get_available_voices();
    json response = {
        {"voices", voices},
        {"model", "kitten-tts"},
        {"sample_rate", g_engine->get_sample_rate()}
    };

    res.set_content(response.dump(), "application/json");
}

// Speech synthesis endpoint
void handle_speech(const httplib::Request& req, httplib::Response& res) {
    if (!g_engine || !g_engine->is_loaded()) {
        res.status = 503;
        res.set_content(R"({"error":"Engine not initialized"})", "application/json");
        return;
    }

    try {
        // Parse request
        json request = json::parse(req.body);

        std::string input = request.value("input", std::string(""));
        std::string voice = request.value("voice", std::string("leo"));
        std::string model = request.value("model", std::string("kitten-tts"));
        std::string response_format = request.value("response_format", std::string("wav"));
        float speed = request.value("speed", 1.0f);

        if (input.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"Missing 'input' field"})", "application/json");
            return;
        }

        // Synthesize speech
        std::vector<float> audio = g_engine->synthesize(input, voice, speed);

        if (audio.empty()) {
            res.status = 500;
            res.set_content(R"({"error":"Synthesis failed"})", "application/json");
            return;
        }

        // Encode to WAV
        std::vector<uint8_t> wav_data = encode_wav(audio, g_engine->get_sample_rate());

        // Set response
        res.set_content(
            reinterpret_cast<const char*>(wav_data.data()),
            wav_data.size(),
            "audio/wav"
        );

    } catch (const json::exception& e) {
        res.status = 400;
        json error = {{"error", "Invalid JSON: " + std::string(e.what())}};
        res.set_content(error.dump(), "application/json");
    } catch (const std::exception& e) {
        res.status = 500;
        json error = {{"error", "Internal error: " + std::string(e.what())}};
        res.set_content(error.dump(), "application/json");
    }
}

// Ollama-compatible voices endpoint
void handle_ollama_voices(const httplib::Request& req, httplib::Response& res) {
    if (!g_engine || !g_engine->is_loaded()) {
        res.status = 503;
        res.set_content(R"({"error":"Engine not initialized"})", "application/json");
        return;
    }

    auto voices = g_engine->get_available_voices();
    json response = json::array();

    for (const auto& voice : voices) {
        response.push_back({
            {"name", voice},
            {"model", "kitten-tts"}
        });
    }

    res.set_content(response.dump(), "application/json");
}

int main(int argc, char* argv[]) {
    std::cout << "KittenTTS Server v1.0" << std::endl;
    std::cout << "====================" << std::endl;

    // Parse command line arguments
    std::string model_dir;
    std::string host = "127.0.0.1";
    int port = 8780;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: kitten-tts-server [model_dir] [options]" << std::endl;
            std::cout << "  model_dir              Directory containing model.onnx and voices.npz" << std::endl;
            std::cout << "  --host HOST            Host to bind to (default: 127.0.0.1)" << std::endl;
            std::cout << "  --port PORT            Port to bind to (default: 8780)" << std::endl;
            std::cout << "  --help, -h             Show this help message" << std::endl;
            return 0;
        } else if (arg[0] != '-') {
            model_dir = arg;
        }
    }

    if (model_dir.empty()) {
        std::cerr << "Error: Model directory required" << std::endl;
        std::cerr << "Usage: kitten-tts-server [model_dir] [--host HOST] [--port PORT]" << std::endl;
        return 1;
    }

    std::cout << "Model directory: " << model_dir << std::endl;
    std::cout << "Host: " << host << std::endl;
    std::cout << "Port: " << port << std::endl;

    // Initialize engine
    std::cout << "Initializing KittenTTS engine..." << std::endl;
    g_engine = std::make_unique<lemon::backends::KittenTtsNative>();

    if (!g_engine->initialize(model_dir)) {
        std::cerr << "Failed to initialize KittenTTS engine" << std::endl;
        return 1;
    }

    std::cout << "Engine initialized successfully!" << std::endl;
    std::cout << "Available voices: ";
    auto voices = g_engine->get_available_voices();
    for (size_t i = 0; i < voices.size(); i++) {
        std::cout << voices[i];
        if (i < voices.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Create HTTP server
    httplib::Server server;

    // Register routes
    server.Get("/health", handle_health);
    server.Get("/v1/voices", handle_voices);
    server.Get("/voices", handle_ollama_voices);
    server.Post("/v1/audio/speech", handle_speech);
    server.Post("/audio/speech", handle_speech);

    // Start server
    std::cout << "\nStarting server at http://" << host << ":" << port << std::endl;
    std::cout << "Endpoints:" << std::endl;
    std::cout << "  GET  /health          - Health check" << std::endl;
    std::cout << "  GET  /v1/voices       - List available voices" << std::endl;
    std::cout << "  POST /v1/audio/speech - Synthesize speech (OpenAI-compatible)" << std::endl;
    std::cout << "  POST /audio/speech    - Synthesize speech (alias)" << std::endl;
    std::cout << std::endl;

    if (!server.listen(host.c_str(), port)) {
        std::cerr << "Failed to start server on " << host << ":" << port << std::endl;
        return 1;
    }

    return 0;
}
