# AI Fundamentals Demo

An interactive CLI chat tool that runs local AI models and displays real-time performance metrics. Built for learning and demonstrating how AI models work under the hood — tokens, latency, KV cache, throughput, and GPU utilization explained as you chat.

---

## Requirements

- Python 3.10+
- Ollama (installed automatically if missing on macOS/Linux)
- NVIDIA GPU optional — Apple Silicon and CPU are supported

---

## Quick Start

**macOS / Linux**
```bash
git clone https://github.com/mmtxcode/ai-fundamentals-demo.git
cd ai-fundamentals-demo
chmod +x run.sh
./run.sh
```

**Windows**
```bat
git clone https://github.com/mmtxcode/ai-fundamentals-demo.git
cd ai-fundamentals-demo
run.bat
```

The launcher will:
1. Check Python 3.10+ is installed
2. Create a virtual environment (`.venv/`) and install dependencies
3. Install Ollama if it is not already on your system
4. Start Ollama if it is not already running
5. Offer to pull the recommended demo models if any are missing

---

## First Run

On first launch you will be prompted to pull the three recommended demo models:

| Model | Size | Description |
|---|---|---|
| `llama3.2:1b` | ~700 MB | Fastest — minimal VRAM, good baseline |
| `llama3.2:3b` | ~2 GB | Balanced speed and quality |
| `llama3.1:8b` | ~4.7 GB | Strongest reasoning, highest VRAM usage |

These three models share the same architecture (Llama 3) at different sizes, making them ideal for demonstrating how parameter count affects performance.

After pulling, select a model and context window size to begin chatting.

---

## Performance Metrics

Every response displays a metrics panel covering:

### Tokens
The atomic unit models process. Each token is roughly ¾ of a word on average. The count shown reflects what the model actually processed — including chat template tokens added by Ollama, which is why the number is higher than a raw word count.

### Time to First Token (TTFT)
How long before the model starts responding. Driven by **prefill** — processing the entire input prompt before generating the first output token. Longer prompts mean slower TTFT.

### Throughput
Output tokens per second during generation. LLM decode is **memory-bandwidth bound**, not compute bound — the GPU spends most of its time moving model weights from VRAM, not calculating. This is why a smaller model can be 3–5× faster than a larger one on the same GPU.

### KV Cache
The Key-Value cache stores attention states for all tokens in the conversation. It grows linearly with context length and lives in VRAM. The usage bar shows how much of the context window is filled. When the cache is full, Ollama silently drops the oldest tokens.

### Cache Reuse
When you send follow-up messages, Ollama can reuse the KV cache for the shared conversation prefix instead of recomputing it. The number of tokens skipped is shown here, calculated from actual Ollama token counts (not estimates).

### GPU Utilization & VRAM
Shows GPU architecture, memory bandwidth, VRAM usage, and utilization during generation. Memory bandwidth is the key spec for LLM performance — it sets the ceiling on how fast tokens can be generated regardless of compute throughput.

---

## Commands

| Command | Description |
|---|---|
| `/compare` | Run the same prompt across multiple models and see a side-by-side metrics comparison |
| `/model` | Switch to a different installed model |
| `/ctx <n>` | Change the context window size — e.g. `/ctx 8192` |
| `/clear` | Clear conversation history and reset the KV cache |
| `/setup` | Pull the recommended demo models |
| `/concepts` | Show the metric concept reference card |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## Demo Scenarios

### 1. Context window overflow
Set a small context window, have a multi-turn conversation, then ask the model to recall something from the beginning:
```
/ctx 512
```
After a few exchanges the red overflow panel will appear. Ask:
```
What was the first question I asked you?
```
The model will reference the wrong prompt — it silently lost the beginning of the conversation when the context filled up.

Reset with a larger window and repeat to show the fix:
```
/clear
/ctx 4096
```

### 2. TTFT vs prompt length
Compare time to first token between a short and a long prompt. The response length may be similar but TTFT will be noticeably higher for the longer input — prefill cost scales with input length.

### 3. Model size comparison
Use `/compare` to run the same prompt across `llama3.2:1b`, `llama3.2:3b`, and `llama3.1:8b`. The side-by-side table shows how parameter count directly affects speed, VRAM usage, and output quality, with a plain-English explanation of why each model won its best metric.

### 4. KV cache prefix reuse
In a multi-turn conversation watch the **Cache reuse** row grow each turn as Ollama skips recomputing the shared conversation prefix. Use `/clear` to reset and observe the first turn has zero cache reuse.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ollama` | Python client for the Ollama API |
| `rich` | Terminal formatting and panels |
| `pynvml` *(optional)* | NVIDIA GPU utilization and memory stats |

To enable NVIDIA GPU monitoring uncomment `pynvml` in `requirements.txt` and re-run `./run.sh`.

---

## Cisco Intersight Integration (MCP)

The demo includes an MCP server (`intersight_mcp.py`) that connects to Cisco Intersight, letting the model call real infrastructure tools during a conversation. This demonstrates the difference between what a model *knows* (training data) and what it can *do* (live tool access).

### Setup

Create a `.env` file in the project root (see `.env.example`). Three auth methods are supported — priority order is as listed:

**Option 1 — OAuth2 Client Credentials (recommended)**
```bash
INTERSIGHT_CLIENT_ID=your_client_id_here
INTERSIGHT_CLIENT_SECRET=your_client_secret_here
INTERSIGHT_BASE_URL=https://intersight.com
```
Generate these in Intersight → Settings → OAuth2 Applications → Create. The script exchanges them for a bearer token automatically — no key files needed.

**Option 2 — OAuth2 pre-fetched bearer token**
```bash
INTERSIGHT_OAUTH_TOKEN=your_bearer_token_here
INTERSIGHT_BASE_URL=https://intersight.com
```
Useful if a CI/CD pipeline or external system already provides a token.

**Option 3 — HTTP Signature (API key + private key)**
```bash
INTERSIGHT_API_KEY_ID=your_key_id_here
INTERSIGHT_API_SECRET_KEY_FILE=~/.intersight/private-key.pem
INTERSIGHT_BASE_URL=https://intersight.com
```

Toggle tools on/off during chat with the `/tools` command.

### Tool Modes

Two modes control how many Intersight tools are registered:

| Mode | Tools | How to enable |
|---|---|---|
| **Core** (default) | 66 read-only tools | `INTERSIGHT_TOOL_MODE=core` (or unset) |
| **All** | 198 tools (core + CRUD) | `INTERSIGHT_TOOL_MODE=all` |

Set the mode in your `.env` file or as an environment variable before launching:

```bash
INTERSIGHT_TOOL_MODE=all ./run.sh
```

**Core mode** includes read-only queries across Inventory, Alarms, Policies, Pools, Telemetry, Network/Fabric, Hardware/Firmware, Workflows, and Security. Good for demos where you want to show live data retrieval without write risk.

**All mode** adds full CRUD operations: create/update/delete for policies, pools, fabric configurations, server profiles, vNICs, and more. Use this when demonstrating end-to-end automation.

---

## Project Structure

```
ai-fundamentals-demo/
├── chat.py              # Main application
├── intersight_mcp.py    # Cisco Intersight MCP server
├── requirements.txt     # Python dependencies
├── run.sh               # macOS / Linux launcher
├── run.bat              # Windows launcher
├── .env.example         # Intersight credentials template
└── .gitignore
```
