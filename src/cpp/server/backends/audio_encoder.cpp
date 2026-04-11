// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 Lemonade Project

#include "lemon/backends/audio_encoder.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <lemon/utils/aixlog.hpp>

namespace lemon {
namespace backends {

// ============================================================================
// Helper Methods
// ============================================================================

void AudioEncoder::write_u16(std::vector<uint8_t>& out, uint16_t val) {
    out.push_back(val & 0xFF);
    out.push_back((val >> 8) & 0xFF);
}

void AudioEncoder::write_u32(std::vector<uint8_t>& out, uint32_t val) {
    out.push_back(val & 0xFF);
    out.push_back((val >> 8) & 0xFF);
    out.push_back((val >> 16) & 0xFF);
    out.push_back((val >> 24) & 0xFF);
}

// ============================================================================
// WAV Encoding
// ============================================================================

void AudioEncoder::encode_wav(
    const std::vector<float>& audio,
    int sample_rate,
    std::vector<uint8_t>& out_vector
) {
    if (audio.empty()) {
        return;
    }

    // Convert float samples to int16
    std::vector<int16_t> samples(audio.size());
    for (size_t i = 0; i < audio.size(); i++) {
        // Clamp to [-1.0, 1.0] range and convert to int16
        float clamped = std::max(-1.0f, std::min(1.0f, audio[i]));
        samples[i] = static_cast<int16_t>(clamped * 32767.0f);
    }

    // WAV file header constants
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t file_size = 36 + data_size;
    const uint16_t audio_format = 1;  // PCM
    const uint16_t num_channels = 1;  // Mono
    const uint32_t byte_rate = sample_rate * num_channels * 2;  // 16-bit = 2 bytes
    const uint16_t block_align = num_channels * 2;
    const uint16_t bits_per_sample = 16;

    out_vector.reserve(44 + data_size);

    // RIFF header
    out_vector.push_back('R');
    out_vector.push_back('I');
    out_vector.push_back('F');
    out_vector.push_back('F');
    write_u32(out_vector, file_size);
    out_vector.push_back('W');
    out_vector.push_back('A');
    out_vector.push_back('V');
    out_vector.push_back('E');

    // fmt chunk
    out_vector.push_back('f');
    out_vector.push_back('m');
    out_vector.push_back('t');
    out_vector.push_back(' ');
    write_u32(out_vector, 16);  // Subchunk1Size for PCM
    write_u16(out_vector, audio_format);
    write_u16(out_vector, num_channels);
    write_u32(out_vector, static_cast<uint32_t>(sample_rate));
    write_u32(out_vector, byte_rate);
    write_u16(out_vector, block_align);
    write_u16(out_vector, bits_per_sample);

    // data chunk
    out_vector.push_back('d');
    out_vector.push_back('a');
    out_vector.push_back('t');
    out_vector.push_back('a');
    write_u32(out_vector, data_size);

    // Audio data (little-endian int16)
    const uint8_t* sample_bytes = reinterpret_cast<const uint8_t*>(samples.data());
    out_vector.insert(out_vector.end(), sample_bytes, sample_bytes + data_size);
}

// ============================================================================
// MP3 Encoding (Streaming)
// ============================================================================

bool AudioEncoder::encode_mp3_streaming(
    const std::vector<float>& audio,
    int sample_rate,
    httplib::DataSink& sink,
    int bitrate
) {
    // Note: Full MP3 encoding requires libmp3lame or similar library.
    // For now, we fall back to WAV encoding with MP3 MIME type.
    // This is a temporary solution until proper MP3 encoding is added.

    LOG(WARNING, "AudioEncoder") << "MP3 encoding not fully implemented, using WAV fallback" << std::endl;

    // Encode as WAV and send
    std::vector<uint8_t> wav_data;
    encode_wav(audio, sample_rate, wav_data);

    if (!wav_data.empty()) {
        sink.write(reinterpret_cast<const char*>(wav_data.data()), wav_data.size());
        return true;
    }

    return false;
}

// ============================================================================
// Opus Encoding (Streaming) - Not Implemented
// ============================================================================

bool AudioEncoder::encode_opus_streaming(
    const std::vector<float>& audio,
    int sample_rate,
    httplib::DataSink& sink,
    int bitrate
) {
    (void)audio;
    (void)sample_rate;
    (void)sink;
    (void)bitrate;

    // Opus encoding requires libopus encoder library
    // Return false to indicate not supported
    LOG(WARNING, "AudioEncoder") << "Opus encoding not implemented" << std::endl;
    return false;
}

// ============================================================================
// AAC Encoding (Streaming) - Not Implemented
// ============================================================================

bool AudioEncoder::encode_aac_streaming(
    const std::vector<float>& audio,
    int sample_rate,
    httplib::DataSink& sink,
    int bitrate
) {
    (void)audio;
    (void)sample_rate;
    (void)sink;
    (void)bitrate;

    // AAC encoding requires FAAC or similar encoder library
    // Return false to indicate not supported
    LOG(WARNING, "AudioEncoder") << "AAC encoding not implemented" << std::endl;
    return false;
}

// ============================================================================
// PCM Encoding (Streaming)
// ============================================================================

void AudioEncoder::write_pcm_streaming(
    const std::vector<float>& audio,
    httplib::DataSink& sink,
    bool int16
) {
    if (audio.empty()) {
        return;
    }

    if (int16) {
        // Convert float [-1.0, 1.0] to int16 and stream
        std::vector<int16_t> samples(audio.size());
        for (size_t i = 0; i < audio.size(); i++) {
            float clamped = std::max(-1.0f, std::min(1.0f, audio[i]));
            samples[i] = static_cast<int16_t>(clamped * 32767.0f);
        }
        sink.write(reinterpret_cast<const char*>(samples.data()), samples.size() * sizeof(int16_t));
    } else {
        // Stream as float32le (raw float samples)
        sink.write(reinterpret_cast<const char*>(audio.data()), audio.size() * sizeof(float));
    }
}

// ============================================================================
// Format Support Check
// ============================================================================

bool AudioEncoder::is_format_supported(const std::string& format) {
    // Currently supported formats
    if (format == "wav") {
        return true;
    }
    if (format == "pcm" || format == "float") {
        return true;
    }
    // MP3 is "supported" but uses WAV fallback
    if (format == "mp3") {
        return true;
    }
    // Opus and AAC not implemented
    if (format == "opus" || format == "aac") {
        return false;
    }
    return false;
}

// ============================================================================
// MIME Type Lookup
// ============================================================================

std::string AudioEncoder::get_mime_type(const std::string& format) {
    if (format == "wav") {
        return "audio/wav";
    }
    if (format == "mp3") {
        return "audio/mpeg";
    }
    if (format == "opus") {
        return "audio/opus";
    }
    if (format == "aac") {
        return "audio/aac";
    }
    if (format == "pcm" || format == "float") {
        return "audio/pcm";
    }
    // Default to WAV
    return "audio/wav";
}

} // namespace backends
} // namespace lemon
