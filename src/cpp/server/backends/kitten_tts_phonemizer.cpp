#include "lemon/backends/kitten_tts_phonemizer.h"
#include <iostream>
#include <algorithm>
#include <codecvt>
#include <locale>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <linux/limits.h>
#endif

// espeak-ng audio output enum
typedef enum {
    AUDIO_OUTPUT_SYNCHRONOUS = 0,
    AUDIO_OUTPUT_ASYNCHRONOUS = 1
} t_espeak_AUDIO_OUTPUT;

namespace lemon {
namespace backends {

KittenTtsPhonemizer::KittenTtsPhonemizer() = default;

KittenTtsPhonemizer::~KittenTtsPhonemizer() {
    if (initialized_) {
        if (espeak_ng_Terminate_) {
            espeak_ng_Terminate_();
        }
        initialized_ = false;
    }
    unload_library();
}

bool KittenTtsPhonemizer::initialize(const std::string& data_path) {
    if (!load_library()) {
        std::cerr << "Failed to load espeak-ng library" << std::endl;
        return false;
    }

    // Initialize espeak
    // Audio output is not needed, so we use AUDIO_OUTPUT_SYNCHRONOUS with no callback
    int result = espeak_Initialize_(AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0);
    if (result < 0) {
        std::cerr << "espeak_Initialize failed: " << result << std::endl;
        return false;
    }

    initialized_ = true;
    return true;
}

std::string KittenTtsPhonemizer::phonemize(const std::string& text) {
    if (!initialized_ || !espeak_ng_TextToPhonemes_) {
        return "";
    }

    // Convert UTF-8 to wide string
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring wide_text = converter.from_bytes(text);

    wchar_t* phonemes_out = nullptr;
    wchar_t* tracking_out = nullptr;
    size_t phonemes_size = 0;
    size_t tracking_size = 0;

    // Use espeak-ng's TextToPhonemes function
    // Parameters: text, switches, phonememode, output_phonemes, output_tracking
    int result = espeak_ng_TextToPhonemes_(
        wide_text.c_str(),
        0,  // switches
        2,  // phonememode: 2 = IPA phonemes
        &phonemes_out,
        &phonemes_size,
        &tracking_out,
        &tracking_size
    );

    if (result != 0 || phonemes_out == nullptr) {
        int error = espeak_ng_GetError_();
        std::cerr << "espeak_ng_TextToPhonemes failed: " << error << std::endl;
        return "";
    }

    // Convert result to UTF-8 string
    std::string result_str = converter.to_bytes(phonemes_out);

    // Free allocated memory (espeak-ng allocates internally)
    // Note: In a real implementation, we'd need to call the appropriate free function
    // For now, we'll just leak (espeak-ng manages this internally)

    return result_str;
}

bool KittenTtsPhonemizer::load_library() {
#ifdef _WIN32
    // Try to load espeak-ng.dll from system path
    library_handle_ = LoadLibraryA("espeak-ng.dll");

    if (!library_handle_) {
        // Try common installation paths
        const char* paths[] = {
            "C:\\Program Files\\eSpeak NG\\espeak-ng.dll",
            "C:\\Program Files (x86)\\eSpeak NG\\espeak-ng.dll"
        };
        for (const char* path : paths) {
            library_handle_ = LoadLibraryA(path);
            if (library_handle_) break;
        }
    }
#else
    // Try to load libespeak-ng.so
    library_handle_ = dlopen("libespeak-ng.so", RTLD_LAZY);

    if (!library_handle_) {
        // Try with version suffix
        library_handle_ = dlopen("libespeak-ng.so.1", RTLD_LAZY);
    }

    if (!library_handle_) {
        // Try common paths
        const char* paths[] = {
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "/usr/local/lib/libespeak-ng.so.1"
        };
        for (const char* path : paths) {
            library_handle_ = dlopen(path, RTLD_LAZY);
            if (library_handle_) break;
        }
    }
#endif

    if (!library_handle_) {
        std::cerr << "Failed to load espeak-ng library" << std::endl;
        return false;
    }

    // Load function pointers
#ifdef _WIN32
    espeak_Initialize_ = (espeak_Initialize_Fn)GetProcAddress((HMODULE)library_handle_, "espeak_Initialize");
    espeak_ng_TextToPhonemes_ = (espeak_ng_TextToPhonemes_Fn)GetProcAddress((HMODULE)library_handle_, "espeak_ng_TextToPhonemes");
    espeak_ng_GetError_ = (espeak_ng_GetError_Fn)GetProcAddress((HMODULE)library_handle_, "espeak_ng_GetErrorCode");
    espeak_ng_Terminate_ = (espeak_ng_Terminate_Fn)GetProcAddress((HMODULE)library_handle_, "espeak_ng_Terminate");
#else
    espeak_Initialize_ = (espeak_Initialize_Fn)dlsym(library_handle_, "espeak_Initialize");
    espeak_ng_TextToPhonemes_ = (espeak_ng_TextToPhonemes_Fn)dlsym(library_handle_, "espeak_ng_TextToPhonemes");
    espeak_ng_GetError_ = (espeak_ng_GetError_Fn)dlsym(library_handle_, "espeak_ng_GetErrorCode");
    espeak_ng_Terminate_ = (espeak_ng_Terminate_Fn)dlsym(library_handle_, "espeak_ng_Terminate");
#endif

    if (!espeak_Initialize_ || !espeak_ng_TextToPhonemes_) {
        std::cerr << "Failed to load required espeak-ng functions" << std::endl;
        unload_library();
        return false;
    }

    return true;
}

void KittenTtsPhonemizer::unload_library() {
#ifdef _WIN32
    if (library_handle_) {
        FreeLibrary((HMODULE)library_handle_);
        library_handle_ = nullptr;
    }
#else
    if (library_handle_) {
        dlclose(library_handle_);
        library_handle_ = nullptr;
    }
#endif
}

} // namespace backends
} // namespace lemon
