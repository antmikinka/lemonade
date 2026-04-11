#pragma once

#include <string>
#include <vector>

namespace lemon {
namespace backends {

/**
 * Text preprocessing and normalization for KittenTTS.
 * Handles number expansion, punctuation normalization, etc.
 */
class KittenTtsPreprocessor {
public:
    KittenTtsPreprocessor() = default;
    ~KittenTtsPreprocessor() = default;

    // Preprocess text for TTS input
    std::string preprocess(const std::string& text);

    // Tokenize text to input IDs
    std::vector<int> tokenize(const std::string& text);

private:
    // Convert number words
    std::string number_to_words(int num);

    // Expand numbers in text
    std::string expand_numbers(const std::string& text);

    // Character to token ID mapping
    int char_to_token(char c);
};

} // namespace backends
} // namespace lemon
