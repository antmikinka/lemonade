// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 Lemonade Project

#pragma once

#include <httplib.h>
#include <string>
#include <vector>

namespace lemon {
namespace backends {

/**
 * AudioEncoder - Audio encoding utilities for TTS output
 *
 * Provides encoding methods for various audio formats:
 * - WAV: Complete file with header
 * - PCM: Raw audio samples (int16 or float32)
 * - MP3: Streaming MP3 encoding
 * - Opus: Streaming Opus encoding (stub)
 * - AAC: Streaming AAC encoding (stub)
 */
class AudioEncoder {
public:
    /**
     * Encode audio samples to WAV format.
     * Builds a complete WAV file in memory.
     *
     * @param audio Audio samples as float [-1.0, 1.0]
     * @param sample_rate Sample rate in Hz (e.g., 24000)
     * @param out_vector Output vector to store WAV data
     */
    static void encode_wav(
        const std::vector<float>& audio,
        int sample_rate,
        std::vector<uint8_t>& out_vector
    );

    /**
     * Encode audio samples to MP3 format (streaming).
     * Writes encoded data directly to the sink.
     *
     * @param audio Audio samples as float [-1.0, 1.0]
     * @param sample_rate Sample rate in Hz
     * @param sink DataSink for streaming output
     * @param bitrate Bitrate in kbps (default: 128)
     * @return true if successful, false if encoder not available
     */
    static bool encode_mp3_streaming(
        const std::vector<float>& audio,
        int sample_rate,
        httplib::DataSink& sink,
        int bitrate = 128
    );

    /**
     * Encode audio samples to Opus format (streaming).
     * Currently not implemented.
     *
     * @param audio Audio samples as float [-1.0, 1.0]
     * @param sample_rate Sample rate in Hz
     * @param sink DataSink for streaming output
     * @param bitrate Bitrate in kbps (default: 96)
     * @return false (not implemented)
     */
    static bool encode_opus_streaming(
        const std::vector<float>& audio,
        int sample_rate,
        httplib::DataSink& sink,
        int bitrate = 96
    );

    /**
     * Encode audio samples to AAC format (streaming).
     * Currently not implemented.
     *
     * @param audio Audio samples as float [-1.0, 1.0]
     * @param sample_rate Sample rate in Hz
     * @param sink DataSink for streaming output
     * @param bitrate Bitrate in kbps (default: 128)
     * @return false (not implemented)
     */
    static bool encode_aac_streaming(
        const std::vector<float>& audio,
        int sample_rate,
        httplib::DataSink& sink,
        int bitrate = 128
    );

    /**
     * Write raw PCM samples to sink.
     *
     * @param audio Audio samples as float [-1.0, 1.0]
     * @param sink DataSink for streaming output
     * @param int16 If true, convert to int16; otherwise write as float32le
     */
    static void write_pcm_streaming(
        const std::vector<float>& audio,
        httplib::DataSink& sink,
        bool int16 = true
    );

    /**
     * Check if the specified format is supported.
     *
     * @param format Format string (wav, mp3, opus, aac, pcm, float)
     * @return true if format is supported, false otherwise
     */
    static bool is_format_supported(const std::string& format);

    /**
     * Get the MIME type for the specified format.
     *
     * @param format Format string (wav, mp3, opus, aac, pcm, float)
     * @return MIME type string (e.g., "audio/wav")
     */
    static std::string get_mime_type(const std::string& format);

private:
    // Helper to write little-endian uint16
    static void write_u16(std::vector<uint8_t>& out, uint16_t val);

    // Helper to write little-endian uint32
    static void write_u32(std::vector<uint8_t>& out, uint32_t val);
};

} // namespace backends
} // namespace lemon
