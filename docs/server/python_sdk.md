# Python SDK (`lemonade-api`)

The official Python client for Lemonade Server. It provides a type-safe, modular interface to all Lemonade Server endpoints using Pydantic models for requests and responses.

**Contents:**

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Client Reference](#client-reference)
  - [Chat & Completions](#chat--completions) | [Embeddings & Reranking](#embeddings--reranking) | [Audio](#audio) | [Images](#images) | [Models](#models) | [Model Management](#model-management) | [System](#system)
- [Data Models Reference](#data-models-reference)
- [Exception Handling](#exception-handling)
- [Guides](#guides)
  - [Streaming](#streaming) | [RAG Pipeline](#rag-pipeline) | [Audio Processing](#audio-processing) | [Model Lifecycle](#model-lifecycle)

## Installation

```bash
pip install lemonade-api
```

**Requirements:** Python 3.8+, `pydantic>=2.0.0`, `requests`

> **API Versioning:** All Lemonade Server endpoints are available on both `/api/v0/` and `/api/v1/` paths. The SDK uses `/api/v1/` by default. Both versions behave identically -- `v0` is maintained for backward compatibility.

## Quick Start

The SDK supports three integration methods. Use whichever fits your project best.

=== "Lemonade SDK"

    ```python
    from lemonade_api import LemonadeClient, ChatCompletionRequest, Message

    # Initialize the client (defaults to http://localhost:8000)
    client = LemonadeClient(base_url="http://localhost:8000")

    # Create a chat request
    request = ChatCompletionRequest(
        model="Qwen3-0.6B-GGUF",
        messages=[
            Message(role="user", content="What is the meaning of life?")
        ]
    )

    # Get the response
    response = client.chat_completions(request)
    print(response.choices[0].message.content)
    ```

=== "OpenAI SDK"

    ```python
    from openai import OpenAI

    # Configure the client for Lemonade Server
    client = OpenAI(
        base_url="http://localhost:8000/api/v1",
        api_key="lemonade"  # Required by SDK but ignored by the server
    )

    completion = client.chat.completions.create(
        model="Qwen3-0.6B-GGUF",
        messages=[
            {"role": "user", "content": "What is the meaning of life?"}
        ]
    )

    print(completion.choices[0].message.content)
    ```

=== "Direct API (curl)"

    ```bash
    curl http://localhost:8000/api/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen3-0.6B-GGUF",
        "messages": [
          {"role": "user", "content": "What is the meaning of life?"}
        ]
      }'
    ```

## Client Reference

### Initialization

```python
from lemonade_api import LemonadeClient

client = LemonadeClient(base_url="http://localhost:8000")
```

| Parameter | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| `base_url` | `str` | No | `"http://localhost:8000"` | Base URL of the Lemonade Server. |

---

### Chat & Completions

#### `chat_completions`

Generate a model response for a chat conversation.

**Endpoint:** `POST /api/v1/chat/completions`

```python
def chat_completions(
    request: ChatCompletionRequest
) -> Union[ChatCompletionResponse, Generator]
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| **`model`** | `str` | **Yes** | - | Model ID (e.g., `"Qwen3-0.6B-GGUF"`). |
| **`messages`** | `List[Message]` | **Yes** | - | Conversation history. Each message has `role` and `content`. |
| `stream` | `bool` | No | `False` | Enable token-by-token streaming via SSE. |
| `temperature` | `float` | No | `None` | Sampling temperature (0.0 - 2.0). |
| `top_p` | `float` | No | `None` | Nucleus sampling probability. |
| `top_k` | `int` | No | `None` | Top-K sampling. |
| `max_tokens` | `int` | No | `None` | Maximum tokens to generate. Deprecated in favor of `max_completion_tokens`. |
| `max_completion_tokens` | `int` | No | `None` | Maximum tokens to generate. Mutually exclusive with `max_tokens`. |
| `stop` | `str \| List[str]` | No | `None` | Up to 4 stop sequences. |
| `repeat_penalty` | `float` | No | `None` | Repetition penalty (1.0 - 2.0). |
| `tools` | `List[Dict]` | No | `None` | Function calling tool definitions. |
| `logprobs` | `bool` | No | `None` | Return log probabilities. |

**Example:**

```python
from lemonade_api import LemonadeClient, ChatCompletionRequest, Message

client = LemonadeClient()

request = ChatCompletionRequest(
    model="Qwen3-0.6B-GGUF",
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is the capital of France?")
    ],
    temperature=0.7,
    max_completion_tokens=100
)

response = client.chat_completions(request)
print(response.choices[0].message.content)
```

**Response type:** `ChatCompletionResponse` with fields `id`, `object`, `created`, `model`, `choices`, and `usage`.

---

#### `completions`

Generate a completion for a text prompt.

**Endpoint:** `POST /api/v1/completions`

```python
def completions(
    request: CompletionRequest
) -> Union[CompletionResponse, Generator]
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| **`model`** | `str` | **Yes** | - | Model ID. |
| **`prompt`** | `str` | **Yes** | - | The prompt to complete. |
| `stream` | `bool` | No | `False` | Enable streaming. |
| `echo` | `bool` | No | `False` | Echo the prompt in the response. |
| `logprobs` | `int` | No | `None` | Return log probabilities. Only when `stream=False`. |
| `temperature` | `float` | No | `None` | Sampling temperature. |
| `repeat_penalty` | `float` | No | `None` | Repetition penalty. |
| `top_k` | `int` | No | `None` | Top-K sampling. |
| `top_p` | `float` | No | `None` | Nucleus sampling. |
| `max_tokens` | `int` | No | `None` | Maximum tokens to generate. |
| `stop` | `str \| List[str]` | No | `None` | Stop sequences. |

**Example:**

```python
from lemonade_api import CompletionRequest

request = CompletionRequest(
    model="Qwen3-0.6B-GGUF",
    prompt="Once upon a time",
    max_tokens=50
)

response = client.completions(request)
print(response.choices[0].text)
```

---

#### `responses`

Generate responses using the OpenAI Responses API format.

**Endpoint:** `POST /api/v1/responses`

```python
def responses(
    request: ResponsesRequest
) -> Union[Dict[str, Any], Generator]
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| **`input`** | `str \| List[Dict]` | **Yes** | - | Input text or structured messages. |
| **`model`** | `str` | **Yes** | - | Model ID. |
| `stream` | `bool` | No | `False` | Enable streaming. |
| `temperature` | `float` | No | `None` | Sampling temperature. |
| `max_output_tokens` | `int` | No | `None` | Maximum tokens to generate. |
| `top_p` | `float` | No | `None` | Nucleus sampling. |
| `top_k` | `int` | No | `None` | Top-K sampling. |
| `repeat_penalty` | `float` | No | `None` | Repetition penalty. |

**Example:**

```python
from lemonade_api import ResponsesRequest

request = ResponsesRequest(
    input="Explain quantum computing in simple terms",
    model="Qwen3-0.6B-GGUF",
    max_output_tokens=200,
    temperature=0.8
)

response = client.responses(request)
print(response)
```

---

### Embeddings & Reranking

#### `embeddings`

Create embedding vectors for input text.

**Endpoint:** `POST /api/v1/embeddings`

```python
def embeddings(
    request: EmbeddingRequest
) -> EmbeddingResponse
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| **`model`** | `str` | **Yes** | - | Embedding model ID (e.g., `"nomic-embed-text-v1-GGUF"`). |
| **`input`** | `str \| List[str]` | **Yes** | - | Text to embed. |
| `encoding_format` | `str` | No | `"float"` | Format: `"float"` or `"base64"`. |

> **Note:** Only available for models using the `llamacpp` or `flm` recipes. ONNX models (OGA recipes) do not support embeddings.

**Example:**

```python
from lemonade_api import EmbeddingRequest

request = EmbeddingRequest(
    model="nomic-embed-text-v1-GGUF",
    input=["Hello, world!", "How are you?"]
)

response = client.embeddings(request)
print(f"Embedding dimension: {len(response.data[0].embedding)}")
```

**Response type:** `EmbeddingResponse` with fields `object`, `data` (list of `EmbeddingData`), `model`, and `usage`.

---

#### `reranking`

Rerank documents by relevance to a query.

**Endpoint:** `POST /api/v1/reranking`

```python
def reranking(
    request: RerankingRequest
) -> RerankingResponse
```

| Field | Type | Required | Description |
|:---|:---|:---|:---|
| **`model`** | `str` | **Yes** | Reranking model ID (e.g., `"bge-reranker-v2-m3-GGUF"`). |
| **`query`** | `str` | **Yes** | The search query. |
| **`documents`** | `List[str]` | **Yes** | Documents to rerank. |

> **Note:** Only available for models using the `llamacpp` recipe.

**Example:**

```python
from lemonade_api import RerankingRequest

request = RerankingRequest(
    model="bge-reranker-v2-m3-GGUF",
    query="What is machine learning?",
    documents=[
        "Machine learning is a subset of AI.",
        "Python is a programming language.",
        "Neural networks are used in deep learning."
    ]
)

response = client.reranking(request)
for result in sorted(response.results, key=lambda x: x.relevance_score, reverse=True):
    print(f"Document {result.index}: {result.relevance_score:.3f}")
```

**Response type:** `RerankingResponse` with fields `model`, `object`, `results` (list of `RerankingResult`), and `usage`.

> **Note:** Results are returned in their original input order. Sort by `relevance_score` descending on the client side to get ranked order.

---

### Audio

#### `audio_transcriptions`

Transcribe audio to text using Whisper models.

**Endpoint:** `POST /api/v1/audio/transcriptions`

```python
def audio_transcriptions(
    request: AudioTranscriptionRequest
) -> Dict[str, Any]
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| **`model`** | `str` | **Yes** | - | Whisper model ID (e.g., `"Whisper-Tiny"`, `"Whisper-Base"`, `"Whisper-Small"`). |
| **`file`** | `str` | **Yes** | - | Path to the audio file (`.wav`). |
| `language` | `str` | No | `None` | ISO 639-1 language code (e.g., `"en"`, `"es"`, `"fr"`). Auto-detected if not set. |
| `response_format` | `str` | No | `"json"` | Response format: `"json"`, `"text"`, `"srt"`, `"vtt"`. |

**Example:**

```python
from lemonade_api import LemonadeClient, AudioTranscriptionRequest

client = LemonadeClient()

request = AudioTranscriptionRequest(
    model="Whisper-Small",
    file="path/to/meeting.wav",
    language="en"
)

response = client.audio_transcriptions(request)
print(f"Transcript: {response['text']}")
```

---

#### `audio_speech`

Generate speech audio from text using Kokoro TTS models.

**Endpoint:** `POST /api/v1/audio/speech`

```python
def audio_speech(
    request: AudioSpeechRequest
) -> bytes
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| **`model`** | `str` | **Yes** | - | TTS model ID (e.g., `"kokoro-v1"`). |
| **`input`** | `str` | **Yes** | - | Text to synthesize. |
| `voice` | `str` | No | `"shimmer"` | Voice ID. Supports OpenAI voices (`alloy`, `ash`, etc.) and Kokoro voices (`af_sky`, `am_echo`, etc.). |
| `speed` | `float` | No | `1.0` | Speech speed. |
| `response_format` | `str` | No | `"mp3"` | Audio format: `"mp3"`, `"wav"`, `"opus"`, `"pcm"`. |
| `stream_format` | `str` | No | `None` | Set to `"audio"` for streaming PCM output. |

**Example:**

```python
from lemonade_api import AudioSpeechRequest

request = AudioSpeechRequest(
    model="kokoro-v1",
    input="Lemonade Server can speak!",
    voice="af_sky",
    speed=1.0
)

# Returns raw audio bytes
audio_data = client.audio_speech(request)

# Save to file
with open("output.mp3", "wb") as f:
    f.write(audio_data)
```

---

### Images

#### `generate_images`

Generate images from text prompts using Stable Diffusion models.

**Endpoint:** `POST /api/v1/images/generations`

```python
def generate_images(
    request: ImageGenerationRequest
) -> Dict[str, Any]
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| **`model`** | `str` | **Yes** | - | Image model ID (e.g., `"SD-Turbo"`, `"SDXL-Turbo"`). |
| **`prompt`** | `str` | **Yes** | - | Text description of the image to generate. |
| `size` | `str` | No | `"512x512"` | Image dimensions (`WIDTHxHEIGHT`). |
| `n` | `int` | No | `1` | Number of images to generate. |
| `response_format` | `str` | No | `"b64_json"` | Response format (only `b64_json` supported). |
| `steps` | `int` | No | `None` | Inference steps. SD-Turbo works well with 4 steps. |
| `cfg_scale` | `float` | No | `None` | Classifier-free guidance scale. |
| `seed` | `int` | No | `None` | Random seed for reproducibility. |

**Example:**

```python
from lemonade_api import ImageGenerationRequest

request = ImageGenerationRequest(
    model="SD-Turbo",
    prompt="A serene mountain landscape at sunset",
    size="512x512",
    steps=4,
    cfg_scale=1.0
)

response = client.generate_images(request)
# Response contains base64-encoded image data in response["data"][0]["b64_json"]
```

---

### Models

#### `list_models`

List available models.

**Endpoint:** `GET /api/v1/models`

```python
def list_models(show_all: bool = False) -> ModelListResponse
```

| Parameter | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| `show_all` | `bool` | No | `False` | If `True`, returns all models including those not yet downloaded. |

**Example:**

```python
# Show only downloaded models
response = client.list_models()
for model in response.data:
    print(f"{model.id} ({model.size} GB) - Labels: {model.labels}")

# Show all available models (including not-yet-downloaded)
response = client.list_models(show_all=True)
for model in response.data:
    status = "Downloaded" if model.downloaded else "Not downloaded"
    print(f"{model.id}: {status}")
```

---

#### `get_model`

Get detailed information about a specific model.

**Endpoint:** `GET /api/v1/models/{model_id}`

```python
def get_model(model_id: str) -> ModelInfo
```

**Example:**

```python
model = client.get_model("Qwen3-0.6B-GGUF")
print(f"Model: {model.id}")
print(f"Checkpoint: {model.checkpoint}")
print(f"Recipe: {model.recipe}")
print(f"Size: {model.size} GB")
print(f"Labels: {model.labels}")
```

---

### Model Management

#### `pull_model`

Download and install a model. Supports both registered models from the Lemonade registry and custom models from Hugging Face.

**Endpoint:** `POST /api/v1/pull`

```python
def pull_model(
    request: PullModelRequest
) -> Union[Dict[str, Any], Generator]
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| **`model_name`** | `str` | **Yes** | - | Model name. Custom models must use the `user.` namespace prefix. |
| `checkpoint` | `str` | No | `None` | Hugging Face checkpoint (e.g., `"unsloth/Qwen3-0.6B-GGUF:Q4_0"`). Required for custom models. |
| `recipe` | `str` | No | `None` | Backend recipe (e.g., `"llamacpp"`, `"flm"`, `"oga-hybrid"`). Required for custom models. |
| `stream` | `bool` | No | `False` | Stream download progress via SSE. |
| `reasoning` | `bool` | No | `False` | Mark as a reasoning model (adds `reasoning` label). |
| `vision` | `bool` | No | `False` | Mark as a vision model (adds `vision` label). |
| `embedding` | `bool` | No | `False` | Mark as an embedding model (adds `embeddings` label). |
| `reranking` | `bool` | No | `False` | Mark as a reranking model (adds `reranking` label). |
| `mmproj` | `str` | No | `None` | Multimodal projector file for vision models. |

**Example:**

```python
from lemonade_api import PullModelRequest

# Pull a registered model
client.pull_model(PullModelRequest(model_name="Qwen3-0.6B-GGUF"))

# Pull a custom model from Hugging Face with streaming progress
request = PullModelRequest(
    model_name="user.Phi-4-Mini-GGUF",
    checkpoint="unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M",
    recipe="llamacpp",
    stream=True
)

for event in client.pull_model(request):
    if 'percent' in event:
        print(f"Download progress: {event['percent']}%")
```

---

#### `load_model`

Load a model into memory with optional configuration overrides.

**Endpoint:** `POST /api/v1/load`

```python
def load_model(
    request: LoadModelRequest
) -> Dict[str, Any]
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| **`model_name`** | `str` | **Yes** | - | Model to load. |
| `ctx_size` | `int` | No | `None` | Context size override. |
| `llamacpp_backend` | `str` | No | `None` | Backend override: `"vulkan"`, `"rocm"`, `"metal"`, or `"cpu"`. |
| `llamacpp_args` | `str` | No | `None` | Custom arguments for llama-server. |
| `save_options` | `bool` | No | `False` | Save these settings as defaults for this model. |

**Example:**

```python
from lemonade_api import LoadModelRequest

# Basic load
client.load_model(LoadModelRequest(model_name="Qwen3-0.6B-GGUF"))

# Load with custom settings and save them
client.load_model(LoadModelRequest(
    model_name="Qwen3-0.6B-GGUF",
    ctx_size=8192,
    llamacpp_backend="vulkan",
    llamacpp_args="--flash-attn on --no-mmap",
    save_options=True
))
```

---

#### `unload_model`

Unload a model from memory to free system resources.

**Endpoint:** `POST /api/v1/unload`

```python
def unload_model(
    request: Optional[UnloadModelRequest] = None
) -> Dict[str, Any]
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| `model_name` | `str` | No | `None` | Model to unload. If `None`, unloads all loaded models. |

**Example:**

```python
from lemonade_api import UnloadModelRequest

# Unload a specific model
client.unload_model(UnloadModelRequest(model_name="Qwen3-0.6B-GGUF"))

# Unload all models
client.unload_model()
```

---

#### `delete_model`

Delete a model from local storage. Unloads it first if currently loaded.

**Endpoint:** `POST /api/v1/delete`

```python
def delete_model(
    request: DeleteModelRequest
) -> Dict[str, Any]
```

| Field | Type | Required | Description |
|:---|:---|:---|:---|
| **`model_name`** | `str` | **Yes** | Model to delete. |

**Example:**

```python
from lemonade_api import DeleteModelRequest

response = client.delete_model(DeleteModelRequest(model_name="old-model"))
print(response)  # {"status": "success", "message": "Deleted model: old-model"}
```

---

### System

#### `health`

Get server health status and information about loaded models.

**Endpoint:** `GET /api/v1/health`

```python
def health() -> HealthResponse
```

**Example:**

```python
health = client.health()
print(f"Status: {health.status}")
print(f"Active model: {health.model_loaded}")
print(f"All loaded: {[m.model_name for m in health.all_models_loaded]}")
print(f"Max models: {health.max_models}")
```

**Response type:** `HealthResponse` with fields:

| Field | Type | Description |
|:---|:---|:---|
| `status` | `str` | Server status, always `"ok"`. |
| `model_loaded` | `str` | Most recently accessed model name. |
| `all_models_loaded` | `List[HealthModelInfo]` | All currently loaded models with details (name, checkpoint, type, device, recipe, backend_url). |
| `max_models` | `Dict[str, int]` | Maximum models per type (llm, embedding, reranking). |

---

#### `stats`

Get performance statistics from the last inference request.

**Endpoint:** `GET /api/v1/stats`

```python
def stats() -> StatsResponse
```

**Example:**

```python
stats = client.stats()
print(f"Time to first token: {stats.time_to_first_token:.3f}s")
print(f"Tokens/sec: {stats.tokens_per_second:.2f}")
print(f"Input tokens: {stats.input_tokens}")
print(f"Output tokens: {stats.output_tokens}")
```

**Response type:** `StatsResponse` with fields:

| Field | Type | Description |
|:---|:---|:---|
| `time_to_first_token` | `float` | Time in seconds until the first token was generated. |
| `tokens_per_second` | `float` | Generation speed in tokens per second. |
| `input_tokens` | `int` | Number of input tokens processed. |
| `output_tokens` | `int` | Number of output tokens generated. |
| `decode_token_times` | `List[float]` | Per-token decode times (seconds). |
| `prompt_tokens` | `int` | Total prompt tokens including cached tokens. |

---

#### `system_info`

Get detailed system hardware and software information, including device enumeration and recipe support.

**Endpoint:** `GET /api/v1/system-info`

```python
def system_info() -> SystemInfoResponse
```

**Example:**

```python
info = client.system_info()
print(f"OS: {info.os_version}")
print(f"Processor: {info.processor}")
print(f"Memory: {info.physical_memory}")
print(f"Devices: {list(info.devices.keys())}")
print(f"Recipes: {list(info.recipes.keys())}")
```

**Response type:** `SystemInfoResponse` with fields:

| Field | Type | Description |
|:---|:---|:---|
| `os_version` | `str` | Operating system version. |
| `processor` | `str` | CPU model name. |
| `physical_memory` | `str` | Total system RAM. |
| `oem_system` | `str` | System/laptop model (Windows only). |
| `bios_version` | `str` | BIOS information (Windows only). |
| `cpu_max_clock` | `str` | Maximum CPU clock speed (Windows only). |
| `windows_power_setting` | `str` | Current power plan (Windows only). |
| `devices` | `Dict` | Hardware devices: CPU, iGPU, dGPU, NPU. |
| `recipes` | `Dict` | Backend recipes and their support status. |

---

#### `live`

Check if the server is reachable. Returns a simple boolean -- useful for health checks, load balancers, and orchestrators.

**Endpoint:** `GET /live`

```python
def live() -> bool
```

**Example:**

```python
if client.live():
    print("Server is online")
else:
    print("Server is unreachable")
```

---

## Data Models Reference

All request and response models use [Pydantic](https://docs.pydantic.dev/) for validation and serialization.

### Request Models

| Model | Endpoint | Required Fields |
|:---|:---|:---|
| `ChatCompletionRequest` | `POST /api/v1/chat/completions` | `model`, `messages` |
| `CompletionRequest` | `POST /api/v1/completions` | `model`, `prompt` |
| `EmbeddingRequest` | `POST /api/v1/embeddings` | `model`, `input` |
| `RerankingRequest` | `POST /api/v1/reranking` | `model`, `query`, `documents` |
| `ResponsesRequest` | `POST /api/v1/responses` | `input`, `model` |
| `AudioTranscriptionRequest` | `POST /api/v1/audio/transcriptions` | `model`, `file` |
| `AudioSpeechRequest` | `POST /api/v1/audio/speech` | `model`, `input` |
| `ImageGenerationRequest` | `POST /api/v1/images/generations` | `model`, `prompt` |
| `PullModelRequest` | `POST /api/v1/pull` | `model_name` |
| `LoadModelRequest` | `POST /api/v1/load` | `model_name` |
| `UnloadModelRequest` | `POST /api/v1/unload` | *(none -- unloads all if empty)* |
| `DeleteModelRequest` | `POST /api/v1/delete` | `model_name` |

### Response Models

| Model | Returned By | Key Fields |
|:---|:---|:---|
| `ChatCompletionResponse` | `chat_completions()` | `id`, `choices`, `usage` |
| `CompletionResponse` | `completions()` | `id`, `choices`, `usage` |
| `EmbeddingResponse` | `embeddings()` | `data` (list of embedding vectors), `usage` |
| `RerankingResponse` | `reranking()` | `results` (list with `index` and `relevance_score`), `usage` |
| `ModelInfo` | `get_model()` | `id`, `checkpoint`, `recipe`, `size`, `downloaded`, `labels` |
| `ModelListResponse` | `list_models()` | `data` (list of `ModelInfo`) |
| `HealthResponse` | `health()` | `status`, `model_loaded`, `all_models_loaded`, `max_models` |
| `StatsResponse` | `stats()` | `time_to_first_token`, `tokens_per_second`, `decode_token_times` |
| `SystemInfoResponse` | `system_info()` | `os_version`, `processor`, `devices`, `recipes` |

### Supporting Models

| Model | Description |
|:---|:---|
| `Message` | Chat message with `role` (`"user"`, `"assistant"`, `"system"`) and `content`. |
| `ChatCompletionChoice` | Single choice in a chat response, with `message` (or `delta` for streaming) and `finish_reason`. |
| `CompletionChoice` | Single choice in a text completion, with `text` and `finish_reason`. |
| `EmbeddingData` | Single embedding vector with `index` and `embedding` (list of floats). |
| `RerankingResult` | Reranking result with `index` and `relevance_score`. |
| `HealthModelInfo` | Loaded model details: `model_name`, `checkpoint`, `type`, `device`, `recipe`, `backend_url`. |
| `DeviceInfo` | Hardware device: `name`, `available`, `cores`, `threads`, `vram_gb`, `power_mode`. |
| `Usage` | Token usage: `prompt_tokens`, `completion_tokens`, `total_tokens`. |

---

## Exception Handling

All exceptions inherit from `LemonadeError`.

| Exception | Description | Key Attributes |
|:---|:---|:---|
| `LemonadeError` | Base exception for all Lemonade API errors. | - |
| `APIError` | Server returned a 4xx or 5xx error. | `message`, `status_code`, `error_type` |
| `ConnectionError` | Cannot connect to the server. | `message` |
| `ValidationError` | Client-side validation failed (e.g., missing required fields). | `message` |

**Example:**

```python
from lemonade_api import LemonadeClient, ChatCompletionRequest, Message
from lemonade_api.exceptions import APIError, ConnectionError

client = LemonadeClient()

try:
    response = client.chat_completions(ChatCompletionRequest(
        model="nonexistent-model",
        messages=[Message(role="user", content="Hello")]
    ))
except ConnectionError:
    print("Could not connect to Lemonade Server. Is it running?")
except APIError as e:
    print(f"API error {e.status_code}: {e.message}")
```

---

## Guides

### Streaming

For a better user experience with longer generations, enable token-by-token streaming:

```python
from lemonade_api import LemonadeClient, ChatCompletionRequest, Message

client = LemonadeClient()

request = ChatCompletionRequest(
    model="Qwen3-0.6B-GGUF",
    messages=[
        Message(role="user", content="Write a short poem about lemons.")
    ],
    stream=True,
    temperature=0.7
)

print("Assistant: ", end="")
for chunk in client.chat_completions(request):
    content = chunk['choices'][0]['delta'].get('content', '')
    print(content, end="", flush=True)
print()
```

When streaming is enabled, `chat_completions()` and `completions()` return a Python generator that yields parsed SSE events as dictionaries. Each chunk contains a `delta` field with incremental content rather than a complete `message`.

---

### RAG Pipeline

Retrieval-Augmented Generation (RAG) combines embeddings, retrieval, and reranking to produce grounded LLM responses. Lemonade Server supports the full pipeline locally.

#### Step 1: Generate Embeddings

```python
from lemonade_api import LemonadeClient, EmbeddingRequest

client = LemonadeClient()

query = "How do I install Lemonade?"
response = client.embeddings(EmbeddingRequest(
    model="nomic-embed-text-v1-GGUF",
    input=query
))

query_embedding = response.data[0].embedding
print(f"Vector dimension: {len(query_embedding)}")
```

#### Step 2: Retrieve Candidates

Use the embedding vector to search a vector database (Chroma, Qdrant, pgvector, etc.) and retrieve the top matching documents.

```python
# Example: candidates retrieved from a vector database
candidates = [
    "Lemonade is a beverage made from lemons.",
    "To install Lemonade Server, run `pip install lemonade-server`.",
    "Beyonce released an album called Lemonade.",
    "Lemonade Server requires Python 3.8+."
]
```

#### Step 3: Rerank Results

```python
from lemonade_api import RerankingRequest

response = client.reranking(RerankingRequest(
    model="bge-reranker-v2-m3-GGUF",
    query=query,
    documents=candidates
))

# Sort by relevance (highest first)
sorted_results = sorted(response.results, key=lambda x: x.relevance_score, reverse=True)

print("Top Results:")
for result in sorted_results:
    print(f"  [{result.relevance_score:.2f}] {candidates[result.index]}")
```

#### Step 4: Generate Response

```python
from lemonade_api import ChatCompletionRequest, Message

# Use top reranked documents as context
context = "\n".join([candidates[r.index] for r in sorted_results[:2]])

response = client.chat_completions(ChatCompletionRequest(
    model="Qwen3-0.6B-GGUF",
    messages=[
        Message(role="system", content=f"Answer based on this context:\n{context}"),
        Message(role="user", content=query)
    ]
))

print(response.choices[0].message.content)
```

---

### Audio Processing

Lemonade Server supports both Speech-to-Text (transcription) and Text-to-Speech (synthesis) locally.

#### Transcription (STT)

```python
from lemonade_api import LemonadeClient, AudioTranscriptionRequest

client = LemonadeClient()

request = AudioTranscriptionRequest(
    model="Whisper-Small",
    file="path/to/meeting.wav",
    language="en"
)

response = client.audio_transcriptions(request)
print(f"Transcript: {response['text']}")
```

#### Speech Synthesis (TTS)

```python
from lemonade_api import AudioSpeechRequest

request = AudioSpeechRequest(
    model="kokoro-v1",
    input="The quick brown fox jumps over the lazy dog.",
    voice="af_sky",
    speed=1.0
)

audio_data = client.audio_speech(request)

with open("output.mp3", "wb") as f:
    f.write(audio_data)
print("Audio saved to output.mp3")
```

---

### Model Lifecycle

Unlike cloud APIs, Lemonade Server gives you explicit control over the model lifecycle: download, load, use, unload, and delete.

#### Full Example

```python
from lemonade_api import (
    LemonadeClient, PullModelRequest, LoadModelRequest,
    UnloadModelRequest, DeleteModelRequest, ChatCompletionRequest, Message
)

client = LemonadeClient()

# 1. Download a model
print("Downloading model...")
client.pull_model(PullModelRequest(model_name="Qwen3-0.6B-GGUF"))

# 2. Load with custom settings
print("Loading model...")
client.load_model(LoadModelRequest(
    model_name="Qwen3-0.6B-GGUF",
    ctx_size=8192,
    llamacpp_backend="vulkan"
))

# 3. Check health
health = client.health()
print(f"Loaded models: {[m.model_name for m in health.all_models_loaded]}")

# 4. Use the model
response = client.chat_completions(ChatCompletionRequest(
    model="Qwen3-0.6B-GGUF",
    messages=[Message(role="user", content="Hello!")]
))
print(f"Response: {response.choices[0].message.content}")

# 5. Check performance
stats = client.stats()
print(f"Speed: {stats.tokens_per_second:.1f} tokens/sec")

# 6. Unload when done
client.unload_model(UnloadModelRequest(model_name="Qwen3-0.6B-GGUF"))
print("Model unloaded.")
```

<!--Copyright (c) 2025 AMD-->
