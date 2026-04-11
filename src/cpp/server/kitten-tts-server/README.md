# KittenTTS Server

Standalone HTTP server for KittenTTS text-to-speech synthesis.

## Overview

This server provides OpenAI-compatible `/v1/audio/speech` endpoint for KittenTTS models.
It bundles ONNX Runtime for inference and espeak-ng for phonemization.

## Building

### Windows

```batch
scripts\build-kitten-tts.bat
```

### Linux

```bash
scripts/build-kitten-tts.sh
```

## Distribution

The build scripts produce a distributable package:
- Windows: `kitten-tts-server-windows-x86_64.zip`
- Linux: `kitten-tts-server-linux-x86_64.tar.gz`

Upload to HuggingFace:
```bash
huggingface-cli upload lemonade-sdk/kitten-tts kitten-tts-server-windows-x86_64.zip
huggingface-cli upload lemonade-sdk/kitten-tts kitten-tts-server-linux-x86_64.tar.gz
```

## Usage

```bash
kitten-tts-server <model_dir> --host 127.0.0.1 --port 8780
```

### Endpoints

- `GET /health` - Health check
- `GET /v1/voices` - List available voices
- `POST /v1/audio/speech` - Synthesize speech (OpenAI-compatible)
- `POST /audio/speech` - Synthesize speech (alias)

### Example Request

```bash
curl -X POST http://127.0.0.1:8780/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kitten-tts",
    "input": "Hello, world!",
    "voice": "leo",
    "speed": 1.0,
    "response_format": "wav"
  }' \
  --output output.wav
```

## Architecture

The server consists of:
- **HTTP Server** (cpp-httplib) - Handles REST API requests
- **ONNX Model** - KittenTTS model inference
- **Voice Manager** - Loads and manages voice embeddings from NPZ files
- **Phonemizer** - Text-to-phoneme conversion using espeak-ng
- **Preprocessor** - Text normalization and tokenization

## Dependencies

- ONNX Runtime 1.18.0
- espeak-ng 1.50
- cpp-httplib
- nlohmann/json

## License

MIT
