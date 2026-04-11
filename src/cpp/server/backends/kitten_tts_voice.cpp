#include "lemon/backends/kitten_tts_voice.h"
#include "lemon/utils/npz_reader.h"
#include <iostream>
#include <algorithm>

namespace lemon {
namespace backends {

KittenTtsVoiceManager::KittenTtsVoiceManager() = default;

bool KittenTtsVoiceManager::load_voices(const std::string& npz_path) {
    try {
        utils::NpzReader reader(npz_path);
        std::vector<std::string> names = reader.get_names();

        for (const auto& name : names) {
            if (!reader.has_array(name)) continue;

            auto shape = reader.get_shape(name);
            if (shape.size() != 2) {
                std::cerr << "Warning: Unexpected shape for voice " << name << std::endl;
                continue;
            }

            // Voice embeddings should be [1, hidden_dim] or [hidden_dim]
            std::vector<float> embedding = reader.get_array(name);

            if (!embedding.empty()) {
                voices_[name] = embedding;
                voice_names_.push_back(name);
            }
        }

        std::cout << "Loaded " << voices_.size() << " voices from " << npz_path << std::endl;
        return !voices_.empty();

    } catch (const std::exception& e) {
        std::cerr << "Failed to load voices: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::string> KittenTtsVoiceManager::get_available_voices() const {
    std::vector<std::string> result;
    result.reserve(voice_aliases_.size());

    for (const auto& alias : voice_aliases_) {
        // Check if the internal voice name exists
        if (voices_.find(alias.second) != voices_.end()) {
            result.push_back(alias.first);
        }
    }

    // Sort alphabetically
    std::sort(result.begin(), result.end());
    return result;
}

const std::vector<float>* KittenTtsVoiceManager::get_voice(const std::string& name) const {
    // First try direct name
    auto it = voices_.find(name);
    if (it != voices_.end()) {
        return &it->second;
    }

    // Try alias mapping
    auto alias_it = voice_aliases_.find(name);
    if (alias_it != voice_aliases_.end()) {
        it = voices_.find(alias_it->second);
        if (it != voices_.end()) {
            return &it->second;
        }
    }

    return nullptr;
}

bool KittenTtsVoiceManager::has_voice(const std::string& name) const {
    // Check direct name
    if (voices_.find(name) != voices_.end()) {
        return true;
    }

    // Check alias
    auto alias_it = voice_aliases_.find(name);
    if (alias_it != voice_aliases_.end()) {
        return voices_.find(alias_it->second) != voices_.end();
    }

    return false;
}

} // namespace backends
} // namespace lemon
