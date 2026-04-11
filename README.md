# AI Fundamentals Demo

Two standalone CLI demos for learning and presenting AI concepts using local models via Ollama.

---

## Demo 1 — AI Fundamentals Chat (`run.sh`)

An interactive chat that runs local AI models and displays real-time performance metrics. Shows how AI models work under the hood — tokens, latency, KV cache, throughput, and GPU utilization explained as you chat.

### Requirements

- Python 3.10+
- Ollama (installed automatically if missing on macOS/Linux)
- NVIDIA GPU optional — Apple Silicon and CPU are supported

### Quick Start

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

### First Run

On first launch you will be prompted to pull the three recommended demo models:

| Model | Size | Description |
|---|---|---|
| `llama3.2:1b` | ~700 MB | Fastest — minimal VRAM, good baseline |
| `llama3.2:3b` | ~2 GB | Balanced speed and quality |
| `llama3.1:8b` | ~4.7 GB | Strongest reasoning, highest VRAM usage |

These three models share the same architecture (Llama 3) at different sizes, making them ideal for demonstrating how parameter count affects performance.

### Performance Metrics

Every response displays a metrics panel covering:

**Tokens** — The atomic unit models process. Each token is roughly ¾ of a word. The count includes chat template tokens added by Ollama, which is why the number is higher than a raw word count.

**Time to First Token (TTFT)** — How long before the model starts responding. Driven by prefill — processing the entire input before generating the first output token. Longer prompts mean slower TTFT.

**Throughput** — Output tokens per second. LLM decode is memory-bandwidth bound, not compute bound — the GPU spends most of its time moving model weights from VRAM. This is why a smaller model can be 3–5× faster than a larger one on the same GPU.

**KV Cache** — Stores attention states for all tokens in the conversation. Grows linearly with context length. When full, Ollama silently drops the oldest tokens.

**Cache Reuse** — How many tokens Ollama skipped recomputing by reusing the shared conversation prefix from prior turns.

**GPU Utilization & VRAM** — Architecture, memory bandwidth, VRAM usage, and utilization during generation.

### Commands

| Command | Description |
|---|---|
| `/compare` | Run the same prompt across multiple models — side-by-side metrics |
| `/model` | Switch to a different installed model |
| `/ctx <n>` | Change context window size — e.g. `/ctx 8192` |
| `/clear` | Clear conversation history and reset the KV cache |
| `/setup` | Pull the recommended demo models |
| `/concepts` | Show the metric concept reference card |
| `/help` | Show all commands |
| `/quit` | Exit |

### Demo Scenarios

**1. Context window overflow**
Set a small context, have a multi-turn conversation, then ask the model to recall something from the beginning:
```
/ctx 512
```
After a few exchanges the red overflow panel appears. Ask: *"What was the first question I asked?"* — the model will get it wrong because it silently lost the start of the conversation.

**2. TTFT vs prompt length**
Compare TTFT between a short and long prompt. Response length may be similar but TTFT will be noticeably higher for the longer input — prefill cost scales with input length.

**3. Model size comparison**
Use `/compare` to run the same prompt across all three models. The table shows how parameter count affects speed, VRAM, and quality.

**4. KV cache prefix reuse**
In a multi-turn conversation, watch Cache Reuse grow each turn. Use `/clear` to reset and observe the first turn has zero cache reuse.

---

## Demo 2 — Intersight AI Chat (`intersight-chat.sh`)

A focused demo that connects a local LLM to live Cisco Intersight infrastructure data via MCP tool calling. Demonstrates the key AI concept: the difference between what a model *knows* (training data) vs. what it can *do* (live tool access).

### Requirements

- Python 3.10+
- Ollama with at least one model installed
- Cisco Intersight account with API credentials

### Quick Start

```bash
chmod +x intersight-chat.sh

# Add your Intersight credentials to .env (see .env.example)
cp .env.example .env
# edit .env with your credentials

./intersight-chat.sh
```

### Credentials

Three auth methods are supported — priority order is as listed:

**Option 1 — OAuth2 Client Credentials (recommended)**
```bash
INTERSIGHT_CLIENT_ID=your_client_id_here
INTERSIGHT_CLIENT_SECRET=your_client_secret_here
```
Generate in Intersight → Settings → OAuth2 Applications → Create. The script exchanges them for a bearer token automatically.

**Option 2 — OAuth2 pre-fetched bearer token**
```bash
INTERSIGHT_OAUTH_TOKEN=your_bearer_token_here
```

**Option 3 — HTTP Signature (API key + private key)**
```bash
INTERSIGHT_API_KEY_ID=your_key_id_here
INTERSIGHT_API_SECRET_KEY_FILE=~/.intersight/private-key.pem
```

### Tool Modes

| Mode | Tools | How to enable |
|---|---|---|
| **Core** (default) | 66 read-only tools | `INTERSIGHT_TOOL_MODE=core` (or unset) |
| **All** | 198 tools (core + CRUD) | `INTERSIGHT_TOOL_MODE=all` |

```bash
INTERSIGHT_TOOL_MODE=all ./intersight-chat.sh
```

**Core mode** — read-only queries: inventory, alarms, policies, firmware, fabric, telemetry. Safe for demos.

**All mode** — adds CRUD: create/update/delete policies, pools, server profiles, vNICs, fabric config, and more.

### Example Prompts

```
list servers
any critical alarms?
what firmware versions are running?
show me the UCS domains
how many blade servers do we have?
summarize the environment
```

### Commands

| Command | Description |
|---|---|
| `/tools` | List all available Intersight tools |
| `/model` | Switch to a different model |
| `/clear` | Clear conversation history |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## Dependencies

| Package | Purpose |
|---|---|
| `ollama` | Python client for the Ollama API |
| `rich` | Terminal formatting and panels |
| `mcp` | Model Context Protocol (Intersight demo) |
| `intersight` | Cisco Intersight Python SDK |
| `certifi` | SSL certificate bundle (fixes macOS cert issues) |
| `python-dotenv` | Load credentials from `.env` file |
| `pynvml` *(optional)* | NVIDIA GPU utilization stats |

To enable NVIDIA GPU monitoring uncomment `pynvml` in `requirements.txt` and re-run `./run.sh`.

---

## Project Structure

```
ai-fundamentals-demo/
├── chat.py                # AI Fundamentals demo
├── intersight-chat.py     # Intersight AI demo
├── intersight_mcp.py      # Cisco Intersight MCP server (66 / 198 tools)
├── requirements.txt       # Python dependencies
├── run.sh                 # AI Fundamentals launcher (macOS/Linux)
├── run.bat                # AI Fundamentals launcher (Windows)
├── intersight-chat.sh     # Intersight demo launcher (macOS/Linux)
├── .env.example           # Intersight credentials template
└── .gitignore
```
