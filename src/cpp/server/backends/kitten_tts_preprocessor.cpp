#include "lemon/backends/kitten_tts_preprocessor.h"
#include <sstream>
#include <cctype>
#include <regex>
#include <map>

namespace lemon {
namespace backends {

std::string KittenTtsPreprocessor::preprocess(const std::string& text) {
    std::string result = text;

    // Expand numbers
    result = expand_numbers(result);

    // Normalize punctuation
    std::string normalized;
    normalized.reserve(result.size());

    for (char c : result) {
        // Keep alphanumeric and basic punctuation
        if (std::isalnum(static_cast<unsigned char>(c)) ||
            c == ' ' || c == '.' || c == ',' || c == '!' || c == '?' ||
            c == ';' || c == ':' || c == '-' || c == '"' || c == '\'') {
            normalized += c;
        }
    }

    // Collapse multiple spaces
    std::string collapsed;
    collapsed.reserve(normalized.size());
    bool last_was_space = false;

    for (char c : normalized) {
        if (c == ' ') {
            if (!last_was_space) {
                collapsed += c;
                last_was_space = true;
            }
        } else {
            collapsed += c;
            last_was_space = false;
        }
    }

    return collapsed;
}

std::vector<int> KittenTtsPreprocessor::tokenize(const std::string& text) {
    std::vector<int> tokens;
    tokens.reserve(text.size());

    // KittenTTS uses a simple character-level tokenization
    // with special tokens for start/end of sequence
    tokens.push_back(1);  // Start token (SOS)

    for (char c : text) {
        int token = char_to_token(c);
        if (token >= 0) {
            tokens.push_back(token);
        }
    }

    tokens.push_back(2);  // End token (EOS)

    return tokens;
}

std::string KittenTtsPreprocessor::number_to_words(int num) {
    if (num == 0) return "zero";

    static const std::map<int, std::string> ones = {
        {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}, {5, "five"},
        {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"}, {10, "ten"},
        {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
        {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"},
        {19, "nineteen"}
    };

    static const std::map<int, std::string> tens = {
        {20, "twenty"}, {30, "thirty"}, {40, "forty"}, {50, "fifty"},
        {60, "sixty"}, {70, "seventy"}, {80, "eighty"}, {90, "ninety"}
    };

    if (num < 0) {
        return "minus " + number_to_words(-num);
    }

    if (num < 20) {
        return ones.at(num);
    }

    if (num < 100) {
        int t = (num / 10) * 10;
        int o = num % 10;
        std::string result = tens.at(t);
        if (o > 0) {
            result += " " + ones.at(o);
        }
        return result;
    }

    if (num < 1000) {
        int h = num / 100;
        int r = num % 100;
        std::string result = ones.at(h) + " hundred";
        if (r > 0) {
            result += " " + number_to_words(r);
        }
        return result;
    }

    // For larger numbers, just return digits
    return std::to_string(num);
}

std::string KittenTtsPreprocessor::expand_numbers(const std::string& text) {
    std::string result = text;

    // Match integer numbers
    std::regex int_regex("(\\d+)");

    std::string output;
    std::string::const_iterator search_start = result.cbegin();
    std::smatch match;

    while (std::regex_search(search_start, result.cend(), match, int_regex)) {
        // Add text before match
        output.append(search_start, match[0].first);

        // Convert number to words
        int num = std::stoi(match[1].str());
        output += number_to_words(num);

        search_start = match[0].second;
    }

    // Add remaining text
    output.append(search_start, result.cend());

    return output;
}

int KittenTtsPreprocessor::char_to_token(char c) {
    // Simple character to token mapping
    if (c >= 'a' && c <= 'z') {
        return c - 'a' + 3;  // Start after special tokens
    }
    if (c >= 'A' && c <= 'Z') {
        return c - 'A' + 3;
    }
    if (c >= '0' && c <= '9') {
        return c - '0' + 29;
    }

    // Punctuation
    switch (c) {
        case ' ': return 39;
        case '.': return 40;
        case ',': return 41;
        case '!': return 42;
        case '?': return 43;
        case ';': return 44;
        case ':': return 45;
        case '-': return 46;
        case '"': return 47;
        case '\'': return 48;
        default: return -1;  // Unknown character
    }
}

} // namespace backends
} // namespace lemon
