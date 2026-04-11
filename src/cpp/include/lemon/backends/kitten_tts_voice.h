#pragma once

#include <string>
#include <vector>
#include <map>

namespace lemon {
namespace backends {

/**
 * Voice embedding manager for KittenTTS.
 * Loads voice embeddings from NPZ files and maps user-friendly names to internal names.
 */
class KittenTtsVoiceManager {
public:
    KittenTtsVoiceManager();
    ~KittenTtsVoiceManager() = default;

    // Load voices from NPZ file
    bool load_voices(const std::string& npz_path);

    // Get available voice names
    std::vector<std::string> get_available_voices() const;

    // Get voice embedding by name
    const std::vector<float>* get_voice(const std::string& name) const;

    // Check if a voice exists
    bool has_voice(const std::string& name) const;

    // Get sample rate
    int get_sample_rate() const { return sample_rate_; }

private:
    std::map<std::string, std::vector<float>> voices_;
    std::vector<std::string> voice_names_;
    int sample_rate_ = 24000; // KittenTTS uses 24kHz

    // Map user-friendly names to internal names
    std::map<std::string, std::string> voice_aliases_ = {
        {"bella", "expr-voice-2-f"},
        {"jasper", "expr-voice-0-m"},
        {"luna", "expr-voice-1-f"},
        {"bruno", "expr-voice-3-m"},
        {"rosie", "expr-voice-4-f"},
        {"hugo", "expr-voice-5-m"},
        {"kiki", "expr-voice-6-f"},
        {"leo", "expr-voice-7-m"}
    };
};

} // namespace backends
} // namespace lemon
