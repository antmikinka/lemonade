#pragma once

#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace lemon {
namespace backends {

/**
 * Wrapper for espeak-ng phonemization engine.
 * Dynamically loads espeak-ng library for text-to-phoneme conversion.
 */
class KittenTtsPhonemizer {
public:
    KittenTtsPhonemizer();
    ~KittenTtsPhonemizer();

    // Initialize the phonemizer
    bool initialize(const std::string& data_path = "");

    // Convert text to phonemes (IPA format)
    std::string phonemize(const std::string& text);

    // Check if initialized
    bool is_initialized() const { return initialized_; }

private:
    bool initialized_ = false;
    void* library_handle_ = nullptr;

    // espeak-ng function pointers
    using espeak_Initialize_Fn = int(*)(int, int, const char*, int);
    using espeak_SetVoiceByName_Fn = int(*)(const char*, int);
    using espeak_SetVoiceByProperties_Fn = int(*)(void*, int);
    using espeak_Synthesize_Fn = int(*)(const char*, size_t, int);
    using espeak_SetUriCallback_Fn = int(*)(void*, const char*, const char*);
    using espeak_ng_TextToPhonemes_Fn = int(*)(const wchar_t*, int, int, wchar_t**, size_t*, wchar_t**, size_t*);
    using espeak_ng_GetError_Fn = int(*)();
    using espeak_ng_Terminate_Fn = void(*)();

    espeak_Initialize_Fn espeak_Initialize_ = nullptr;
    espeak_ng_TextToPhonemes_Fn espeak_ng_TextToPhonemes_ = nullptr;
    espeak_ng_GetError_Fn espeak_ng_GetError_ = nullptr;
    espeak_ng_Terminate_Fn espeak_ng_Terminate_ = nullptr;

    bool load_library();
    void unload_library();
};

} // namespace backends
} // namespace lemon
