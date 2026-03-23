# Using Lemonade with Brave Leo BYOM

This guide shows how to configure [Brave Leo](https://brave.com/leo) to use Lemonade as a local AI backend via the **Bring Your Own Model (BYOM)** feature.

## Overview

Lemonade provides an OpenAI-compatible API that can serve as a local AI backend for Brave Leo. This allows you to:

- Run AI models locally on your hardware (GPU/NPU)
- Keep your data private - no data sent to external APIs
- Use Lemonade's model catalog with Brave Leo

## Prerequisites

1. **Lemonade installed and running**
   - Download from: https://github.com/lemonade-sdk/lemonade
   - Start the server: `lemonade-server serve`
   - Verify API is accessible: `curl http://localhost:8000/v1/models`

2. **Brave Browser with Leo**
   - Brave Browser version 1.69 or higher
   - Leo AI feature enabled

## Configuration Steps

### Step 1: Start Lemonade Server

```bash
# Start Lemonade with your preferred model
lemonade-server serve --port 8000
```

The server will be available at `http://localhost:8000`

### Step 2: Configure Brave Leo BYOM

1. Open Brave Browser Settings
2. Navigate to **Leo** settings
3. Scroll to **Bring your own model** section
4. Click **Add new model**
5. Fill in the configuration:

| Field | Value |
|-------|-------|
| **Label** | `Lemonade Local` |
| **Model request name** | `<model-name>` (see below) |
| **Server endpoint** | `http://localhost:8000/v1/chat/completions` |
| **API Key** | (leave blank for local server) |

### Step 3: Available Models

Check available models in Lemonade:

```bash
curl http://localhost:8000/v1/models | jq '.data[] | {id: .id, labels: .labels}'
```

Example model names to use in Brave Leo:
- `Llama-3.2-1B-Instruct-GGUF` - Lightweight chat model
- `Llama-3.2-3B-Instruct-GGUF` - Balanced chat model
- `Qwen2.5-7B-Instruct-GGUF` - Powerful chat model
- `gpt-oss-20b-mxfp4-GGUF` - Large reasoning model

### Step 4: Test the Connection

1. Select your Lemonade model in Brave Leo's model selector
2. Ask a simple question
3. Verify you get a response from your local Lemonade server

## Example Configuration

```json
{
  "label": "Lemonade Llama-3.2",
  "model": "Llama-3.2-3B-Instruct-GGUF",
  "endpoint": "http://localhost:8000/v1/chat/completions",
  "api_key": ""
}
```

## Troubleshooting

### "Connection refused" error
- Ensure Lemonade server is running: `lemonade-server status`
- Check the port is correct (default: 8000)
- Verify firewall isn't blocking localhost connections

### "Model not found" error
- Check the model name matches exactly (case-sensitive)
- Ensure the model is loaded: `lemonade-server list`
- Load the model if needed: `lemonade-server pull <model-name>`

### Slow responses
- Check system resources (RAM, GPU/NPU utilization)
- Try a smaller model (e.g., Llama-3.2-1B vs 3B)
- Monitor Lemonade logs: `lemonade-server logs`

## Privacy Notes

When using Lemonade as your BYOM backend:
- All processing happens locally on your machine
- No data is sent to external APIs
- Brave's reverse proxy is bypassed - direct localhost connection
- Model responses stay on your device

## Advanced: API Key Authentication

If you've set `LEMONADE_API_KEY` environment variable:

```bash
export LEMONADE_API_KEY=your-secret-key
lemonade-server serve
```

Then enter the same key in Brave Leo's BYOM configuration.

## See Also

- [Lemonade Documentation](https://github.com/lemonade-sdk/lemonade)
- [Brave Leo BYOM Guide](https://support.brave.com/hc/en-us/articles/18341176818708-How-do-I-use-the-Bring-Your-Own-Model-BYOM-with-Brave-Leo-)
- [Available Models](https://github.com/lemonade-sdk/lemonade/blob/main/src/cpp/resources/server_models.json)
