"""
KittenTTS integration tests for Lemonade Server.

Tests the /audio/speech endpoint for KittenTTS models.
Covers:
- Basic audio/speech endpoint functionality
- Model loading/unloading
- Voice selection (all 8 KittenTTS voices)
- Format support (wav, pcm, mp3)
- Speed control
- Error cases (model not loaded, invalid format, empty input)

Usage:
    python test/server_kitten_tts.py
    python test/server_kitten_tts.py --server-per-test
    python test/server_kitten_tts.py --server-binary /path/to/lemonade-server
    python test/server_kitten_tts.py --wrapped-server kitten-tts
"""

import base64
import requests
import pytest

from utils.server_base import (
    ServerTestBase,
    run_server_tests,
)
from utils.test_models import (
    PORT,
    TIMEOUT_MODEL_OPERATION,
    TIMEOUT_DEFAULT,
)

# KittenTTS test configuration
KITTEN_TTS_MODEL = "kitten-tts-mini"
KITTEN_TTS_VOICES = ["bella", "jasper", "luna", "bruno", "rosie", "hugo", "kiki", "leo"]
SUPPORTED_FORMATS = ["wav", "mp3", "pcm"]


class KittenTTSTests(ServerTestBase):
    """Tests for KittenTTS Text to Speech."""

    def test_001_basic_tts(self):
        """Test basic speech generation with KittenTTS."""
        payload = {
            "model": KITTEN_TTS_MODEL,
            "input": "Hello, this is a test of KittenTTS.",
            "response_format": "mp3",
        }

        print(f"[INFO] Sending speech generation request with model {KITTEN_TTS_MODEL}")

        response = requests.post(
            f"{self.base_url}/audio/speech",
            json=payload,
            timeout=TIMEOUT_MODEL_OPERATION,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Speech generation failed with status {response.status_code}: {response.text}",
        )

        # MP3 files start with ID3 tag or valid frame sync
        content = response.content
        self.assertTrue(
            len(content) > 0,
            "Response content should not be empty",
        )

        print(f"[OK] Basic TTS test successful - generated {len(content)} bytes")

    def test_002_all_voices(self):
        """Test all 8 KittenTTS voices."""
        for voice in KITTEN_TTS_VOICES:
            with self.subTest(voice=voice):
                payload = {
                    "model": KITTEN_TTS_MODEL,
                    "input": f"Testing voice: {voice}",
                    "voice": voice,
                    "response_format": "wav",
                }

                response = requests.post(
                    f"{self.base_url}/audio/speech",
                    json=payload,
                    timeout=TIMEOUT_MODEL_OPERATION,
                )

                self.assertEqual(
                    response.status_code,
                    200,
                    f"Voice {voice} failed with status {response.status_code}: {response.text}",
                )

                # WAV files start with RIFF header
                self.assertTrue(
                    response.content[:4] == b"RIFF",
                    f"Voice {voice}: Response should be a valid WAV file",
                )

                print(f"[OK] Voice {voice} successful")

    def test_003_format_support(self):
        """Test supported audio formats."""
        for fmt in SUPPORTED_FORMATS:
            with self.subTest(format=fmt):
                payload = {
                    "model": KITTEN_TTS_MODEL,
                    "input": "Testing format support",
                    "response_format": fmt,
                }

                response = requests.post(
                    f"{self.base_url}/audio/speech",
                    json=payload,
                    timeout=TIMEOUT_MODEL_OPERATION,
                )

                self.assertEqual(
                    response.status_code,
                    200,
                    f"Format {fmt} failed with status {response.status_code}: {response.text}",
                )

                # Verify non-empty response
                self.assertTrue(
                    len(response.content) > 0,
                    f"Format {fmt}: Response should not be empty",
                )

                print(f"[OK] Format {fmt} successful")

    def test_004_speed_control(self):
        """Test speed control (0.5x to 2.0x)."""
        test_speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        for speed in test_speeds:
            with self.subTest(speed=speed):
                payload = {
                    "model": KITTEN_TTS_MODEL,
                    "input": "Testing speed control",
                    "speed": speed,
                    "response_format": "wav",
                }

                response = requests.post(
                    f"{self.base_url}/audio/speech",
                    json=payload,
                    timeout=TIMEOUT_MODEL_OPERATION,
                )

                self.assertEqual(
                    response.status_code,
                    200,
                    f"Speed {speed} failed with status {response.status_code}: {response.text}",
                )

                print(f"[OK] Speed {speed} successful")

    def test_005_missing_input_error(self):
        """Test error handling when input is missing."""
        payload = {
            "model": KITTEN_TTS_MODEL,
            # No input field
        }

        response = requests.post(
            f"{self.base_url}/audio/speech",
            json=payload,
            timeout=TIMEOUT_DEFAULT,
        )

        # Should return an error (400 or 422)
        self.assertIn(
            response.status_code,
            [400, 422],
            f"Expected 400 or 422 for missing input, got {response.status_code}",
        )

        # Verify error response structure
        error_data = response.json()
        self.assertIn("error", error_data, "Response should contain error field")

        print(f"[OK] Correctly rejected request without input: {response.status_code}")

    def test_006_empty_input_error(self):
        """Test error handling with empty input string."""
        payload = {
            "model": KITTEN_TTS_MODEL,
            "input": "",
        }

        response = requests.post(
            f"{self.base_url}/audio/speech",
            json=payload,
            timeout=TIMEOUT_DEFAULT,
        )

        # Should return an error for empty input
        self.assertIn(
            response.status_code,
            [400, 422],
            f"Expected 400 or 422 for empty input, got {response.status_code}",
        )

        print(f"[OK] Correctly rejected empty input: {response.status_code}")

    def test_007_invalid_format_error(self):
        """Test error handling with unsupported format."""
        payload = {
            "model": KITTEN_TTS_MODEL,
            "input": "Testing invalid format",
            "response_format": "invalid_format_xyz",
        }

        response = requests.post(
            f"{self.base_url}/audio/speech",
            json=payload,
            timeout=TIMEOUT_DEFAULT,
        )

        # Should return an error for unsupported format
        self.assertIn(
            response.status_code,
            [400, 422, 500],
            f"Expected error for invalid format, got {response.status_code}",
        )

        print(f"[OK] Correctly rejected invalid format: {response.status_code}")

    def test_008_invalid_model_error(self):
        """Test error handling with invalid model."""
        payload = {
            "model": "kitten-tts-nonexistent-model",
            "input": "Testing invalid model",
        }

        response = requests.post(
            f"{self.base_url}/audio/speech",
            json=payload,
            timeout=TIMEOUT_DEFAULT,
        )

        # Should return an error (model not found or similar)
        self.assertIn(
            response.status_code,
            [400, 404, 422, 500],
            f"Expected error for invalid model, got {response.status_code}",
        )

        print(f"[OK] Correctly rejected invalid model: {response.status_code}")

    def test_009_model_loading(self):
        """Test model loading via /api/v1/load endpoint."""
        # First unload the model if loaded
        unload_payload = {"model": KITTEN_TTS_MODEL}
        requests.post(
            f"{self.base_url}/unload",
            json=unload_payload,
            timeout=TIMEOUT_DEFAULT,
        )

        # Load the model
        load_payload = {
            "model": KITTEN_TTS_MODEL,
            "recipe": "kitten-tts",
        }

        load_response = requests.post(
            f"{self.base_url}/load",
            json=load_payload,
            timeout=TIMEOUT_MODEL_OPERATION,
        )

        self.assertEqual(
            load_response.status_code,
            200,
            f"Model load failed with status {load_response.status_code}: {load_response.text}",
        )

        print(f"[OK] Model {KITTEN_TTS_MODEL} loaded successfully")

        # Verify model works after loading
        speech_payload = {
            "model": KITTEN_TTS_MODEL,
            "input": "Model loaded successfully",
            "response_format": "wav",
        }

        speech_response = requests.post(
            f"{self.base_url}/audio/speech",
            json=speech_payload,
            timeout=TIMEOUT_MODEL_OPERATION,
        )

        self.assertEqual(
            speech_response.status_code,
            200,
            f"Speech generation after load failed: {speech_response.text}",
        )

        print(f"[OK] Speech generation successful after explicit load")

    def test_010_model_unloading(self):
        """Test model unloading via /api/v1/unload endpoint."""
        # First ensure model is loaded
        load_payload = {
            "model": KITTEN_TTS_MODEL,
            "recipe": "kitten-tts",
        }
        requests.post(
            f"{self.base_url}/load",
            json=load_payload,
            timeout=TIMEOUT_MODEL_OPERATION,
        )

        # Unload the model
        unload_payload = {"model": KITTEN_TTS_MODEL}
        unload_response = requests.post(
            f"{self.base_url}/unload",
            json=unload_payload,
            timeout=TIMEOUT_DEFAULT,
        )

        self.assertEqual(
            unload_response.status_code,
            200,
            f"Model unload failed with status {unload_response.status_code}: {unload_response.text}",
        )

        print(f"[OK] Model {KITTEN_TTS_MODEL} unloaded successfully")

        # Verify model is unloaded - next request should trigger load or fail
        speech_payload = {
            "model": KITTEN_TTS_MODEL,
            "input": "Testing after unload",
            "response_format": "wav",
        }

        # After unload, server should either auto-load or return error
        speech_response = requests.post(
            f"{self.base_url}/audio/speech",
            json=speech_payload,
            timeout=TIMEOUT_MODEL_OPERATION,
        )

        # Accept either success (auto-load) or specific error
        self.assertIn(
            speech_response.status_code,
            [200, 400, 500],
            f"Unexpected status after unload: {speech_response.status_code}",
        )

        print(f"[OK] Post-unload behavior verified: {speech_response.status_code}")

    def test_011_quad_prefix_endpoints(self):
        """Test that audio/speech is available under all quad-prefix routes."""
        prefixes = [
            "/api/v1",
            "/api/v0",
            "/v1",
            "/v0",
        ]

        payload = {
            "model": KITTEN_TTS_MODEL,
            "input": "Testing endpoint prefixes",
            "response_format": "wav",
        }

        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                response = requests.post(
                    f"http://localhost:{PORT}{prefix}/audio/speech",
                    json=payload,
                    timeout=TIMEOUT_MODEL_OPERATION,
                )

                self.assertEqual(
                    response.status_code,
                    200,
                    f"Endpoint {prefix}/audio/speech failed with status {response.status_code}",
                )

                print(f"[OK] Endpoint {prefix}/audio/speech successful")

    def test_012_long_text_synthesis(self):
        """Test synthesis with longer text (automatic chunking)."""
        long_text = """
        This is a longer text to test the automatic chunking feature of KittenTTS.
        The system should handle this by breaking the text into smaller chunks
        and synthesizing each chunk separately before combining them.
        This ensures that even very long texts can be processed without running
        into memory limitations or quality degradation.
        """ * 3  # Repeat to ensure sufficient length

        payload = {
            "model": KITTEN_TTS_MODEL,
            "input": long_text,
            "response_format": "wav",
        }

        response = requests.post(
            f"{self.base_url}/audio/speech",
            json=payload,
            timeout=TIMEOUT_MODEL_OPERATION,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Long text synthesis failed with status {response.status_code}: {response.text}",
        )

        # Long text should produce larger output
        self.assertTrue(
            len(response.content) > 1000,
            f"Long text should produce substantial audio output, got {len(response.content)} bytes",
        )

        print(f"[OK] Long text synthesis successful - {len(response.content)} bytes")

    def test_013_voice_not_found_error(self):
        """Test error handling with non-existent voice."""
        payload = {
            "model": KITTEN_TTS_MODEL,
            "input": "Testing invalid voice",
            "voice": "nonexistent_voice_xyz",
            "response_format": "wav",
        }

        response = requests.post(
            f"{self.base_url}/audio/speech",
            json=payload,
            timeout=TIMEOUT_MODEL_OPERATION,
        )

        # Server should either error or fall back to default voice
        self.assertIn(
            response.status_code,
            [200, 400, 422],
            f"Unexpected status for invalid voice: {response.status_code}",
        )

        # If successful, should use default voice (bella)
        if response.status_code == 200:
            self.assertTrue(
                len(response.content) > 0,
                "Response should contain audio data",
            )
            print("[OK] Invalid voice handled (used default or returned error)")
        else:
            print(f"[OK] Invalid voice rejected with status {response.status_code}")


if __name__ == "__main__":
    run_server_tests(
        KittenTTSTests,
        "KITTEN TTS INTEGRATION TESTS",
        wrapped_server="kitten-tts",
        modality="tts",
        default_wrapped_server="kitten-tts",
    )
