#!/usr/bin/env python3
"""
NVIDIA Inference Demo — Interactive CLI for LLM inference on NVIDIA GPUs
========================================================================
Educational tool showing how AI inference works on NVIDIA hardware (L40S and
similar Ada Lovelace / Hopper GPUs).

Connects to any OpenAI-compatible inference server running locally:
  • vLLM          — most common open-source serving stack
  • NVIDIA NIM    — NVIDIA's containerized inference microservices
  • TensorRT-LLM  — NVIDIA's high-performance inference library

Environment variables (optional — all have sensible defaults):
  INFERENCE_BASE_URL   Inference server base URL  (default: http://localhost:8000/v1)
  INFERENCE_API_KEY    API key if required         (default: "token-abc123")
  INFERENCE_MODEL      Model name to use           (default: auto-detected)

Run:  ./nvidia-run.sh   (sets up venv and launches)
      python nvidia-inference-demo.py
"""

import os
import sys
import time
import threading
import subprocess
import platform
import json as _json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    import openai
except ImportError:
    print("Missing dependency: openai. Run: pip install openai rich pynvml")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.markup import escape
    from rich.columns import Columns
    from rich.layout import Layout
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
except ImportError:
    print("Missing dependency: rich. Run: pip install openai rich pynvml")
    sys.exit(1)

console = Console()

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL   = os.environ.get("INFERENCE_BASE_URL", "http://localhost:8000/v1")
API_KEY    = os.environ.get("INFERENCE_API_KEY",  "token-abc123")
MODEL_HINT = os.environ.get("INFERENCE_MODEL",    "")

# L40S hardware reference (Ada Lovelace, 48 GB GDDR6)
L40S_SPECS = {
    "architecture":     "Ada Lovelace (sm_89)",
    "vram_gb":          48,
    "memory_bw_gbs":    864,
    "fp16_tflops":      362.1,
    "int8_tops":        724.2,
    "int4_tops":        1448.4,
    "tensor_cores":     "4th-gen (FP8 / FP16 / BF16 / INT8 / INT4)",
    "nvlink":           "No (PCIe)",
    "sm_count":         142,
    "cuda_cores":       18176,
}


# ── CUDA architecture helper ──────────────────────────────────────────────────

def _cuda_arch_name(major: int, minor: int) -> str:
    mapping = {
        (9, 0): "Hopper",
        (8, 9): "Ada Lovelace",
        (8, 6): "Ampere",
        (8, 0): "Ampere",
        (7, 5): "Turing",
        (7, 0): "Volta",
        (6, 1): "Pascal",
    }
    return mapping.get((major, minor), f"sm_{major}{minor}")


# ── GPU Monitor ───────────────────────────────────────────────────────────────

class GPUMonitor:
    """
    Samples GPU utilization, memory, temperature, and clock speeds via pynvml.

    Key concepts for NVIDIA inference:
      • SM Utilization — how busy the CUDA shader multiprocessors are.
        LLM decode is memory-bandwidth bound, so util is often surprisingly low
        even when the GPU is working hard.

      • Memory bandwidth — the true bottleneck. Each forward pass streams all
        model weights from GDDR6 into the SMs. Throughput ∝ BW / model_bytes.

      • Tensor core utilization — requires DCGM profiling metrics, not exposed
        by nvml directly. High SM util + fast tok/s is a proxy.
    """

    def __init__(self):
        self.samples_util:     list[float] = []
        self.samples_mem_gb:   list[float] = []
        self.samples_temp_c:   list[float] = []
        self.samples_sm_clock: list[int]   = []
        self.running   = False
        self._thread:  threading.Thread | None = None
        self.available = False
        self.backend   = None
        self._handle   = None
        self.gpu_name  = "N/A"
        self.total_mem_gb = 0.0
        self.specs: dict = {}
        self._setup()

    def _setup(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.gpu_name     = name if isinstance(name, str) else name.decode()
            self.total_mem_gb = info.total / 1024**3
            self.available    = True
            self.backend      = "pynvml"
            self.gpu_count    = count
            self._load_specs_pynvml()
            return
        except Exception:
            pass

        # Fall back to nvidia-smi
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if r.returncode == 0:
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                self.gpu_name     = parts[0]
                self.total_mem_gb = float(parts[1]) / 1024
                self.available    = True
                self.backend      = "nvidia-smi"
                self.gpu_count    = 1
                self.specs = {
                    "driver": parts[2] if len(parts) > 2 else "?",
                    "compute_cap": parts[3] if len(parts) > 3 else "?",
                }
                cap = self.specs.get("compute_cap", "")
                try:
                    maj, mn = int(cap.split(".")[0]), int(cap.split(".")[1])
                    self.specs["architecture"] = _cuda_arch_name(maj, mn)
                except Exception:
                    pass
                return
        except Exception:
            pass

        self.available = False
        self.gpu_name  = "No NVIDIA GPU detected"
        self.gpu_count = 0

    def _load_specs_pynvml(self):
        try:
            import pynvml
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(self._handle)
            self.specs["compute_cap"]   = f"{major}.{minor}"
            self.specs["architecture"]  = _cuda_arch_name(major, minor)
            mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(self._handle, pynvml.NVML_CLOCK_MEM)
            bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(self._handle)
            self.specs["mem_bw_gbs"]    = round(mem_clock * 2 * bus_width / 8 / 1000, 1)
            self.specs["bus_width"]     = bus_width
            driver = pynvml.nvmlSystemGetDriverVersion()
            self.specs["driver"]        = driver if isinstance(driver, str) else driver.decode()
            self.specs["sm_count"]      = pynvml.nvmlDeviceGetNumGpuCores(self._handle)
        except Exception:
            pass

    def _sample(self):
        if self.backend == "pynvml":
            import pynvml
            info  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util  = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = 0
            try:
                clk = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
            except Exception:
                clk = 0
            return float(util.gpu), info.used / 1024**3, float(temp), clk

        elif self.backend == "nvidia-smi":
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            parts = r.stdout.strip().split(", ")
            return (float(parts[0]), float(parts[1]) / 1024,
                    float(parts[2]), int(parts[3]))

        return 0.0, 0.0, 0.0, 0

    def start(self):
        for lst in (self.samples_util, self.samples_mem_gb,
                    self.samples_temp_c, self.samples_sm_clock):
            lst.clear()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while self.running:
            try:
                u, m, t, c = self._sample()
                self.samples_util.append(u)
                self.samples_mem_gb.append(m)
                self.samples_temp_c.append(t)
                self.samples_sm_clock.append(c)
            except Exception:
                pass
            time.sleep(0.35)

    def stats(self) -> dict | None:
        if not self.available or not self.samples_util:
            return None
        return {
            "name":         self.gpu_name,
            "avg_util":     sum(self.samples_util)     / len(self.samples_util),
            "peak_util":    max(self.samples_util),
            "avg_mem_gb":   sum(self.samples_mem_gb)   / len(self.samples_mem_gb),
            "peak_mem_gb":  max(self.samples_mem_gb),
            "total_mem_gb": self.total_mem_gb,
            "avg_temp_c":   sum(self.samples_temp_c)   / len(self.samples_temp_c),
            "peak_temp_c":  max(self.samples_temp_c),
            "avg_sm_clk":   int(sum(self.samples_sm_clock) / len(self.samples_sm_clock))
                            if self.samples_sm_clock else 0,
        }

    def snapshot(self) -> dict:
        """Instant single-sample — used for idle display."""
        if not self.available:
            return {}
        try:
            u, m, t, c = self._sample()
            return {
                "util":     u,
                "mem_gb":   m,
                "temp_c":   t,
                "sm_clk":   c,
                "total_gb": self.total_mem_gb,
            }
        except Exception:
            return {}


# ── Inference Metrics ─────────────────────────────────────────────────────────

class InferenceMetrics:
    """Timing and token counts for one chat turn."""

    def __init__(self):
        self.t_request:          float = 0.0
        self.t_first_token:      float = 0.0
        self.t_done:             float = 0.0
        self.first_token_received = False
        self.prompt_tokens:      int   = 0
        self.output_tokens:      int   = 0
        self.finish_reason:      str   = ""

    @property
    def ttft(self) -> float:
        if self.t_first_token and self.t_request:
            return self.t_first_token - self.t_request
        return 0.0

    @property
    def total_wall(self) -> float:
        return self.t_done - self.t_request if self.t_done else 0.0

    @property
    def decode_secs(self) -> float:
        """Wall time spent in decode phase (after first token)."""
        if self.t_done and self.t_first_token:
            return self.t_done - self.t_first_token
        return 0.0

    @property
    def generation_tps(self) -> float:
        if self.decode_secs > 0 and self.output_tokens > 1:
            return (self.output_tokens - 1) / self.decode_secs
        return 0.0

    @property
    def prefill_tps(self) -> float:
        if self.ttft > 0 and self.prompt_tokens > 0:
            return self.prompt_tokens / self.ttft
        return 0.0


# ── Rendering ─────────────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    n = max(0, min(width, int(pct / 100 * width)))
    return fill * n + empty * (width - n)


def _color_pct(pct: float, warn: float = 70, danger: float = 90) -> str:
    if pct >= danger:
        return "red"
    if pct >= warn:
        return "yellow"
    return "green"


def render_gpu_header(gpu: GPUMonitor) -> Panel:
    """
    One-time hardware banner explaining the GPU capabilities relevant
    to LLM inference.
    """
    specs = gpu.specs
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("K", style="bold cyan", width=28)
    t.add_column("V", style="white")
    t.add_column("Note", style="dim")

    arch = specs.get("architecture", "?")
    cap  = specs.get("compute_cap",  "?")
    t.add_row("GPU", f"[bold green]{gpu.gpu_name}[/]",
              f"CUDA compute capability {cap}")
    t.add_row("Architecture", arch,
              "Ada Lovelace = sm_89; Hopper = sm_90")
    t.add_row("VRAM", f"{gpu.total_mem_gb:.0f} GB",
              "Model weights + KV cache live here")

    bw = specs.get("mem_bw_gbs")
    if bw:
        t.add_row("Memory Bandwidth", f"{bw} GB/s",
                  "Decode throughput ∝ BW / bytes-per-token")

    if specs.get("sm_count"):
        t.add_row("SM Count", str(specs["sm_count"]),
                  "Streaming Multiprocessors — parallel execution units")

    t.add_row("Driver", specs.get("driver", "?"), "")
    t.add_row("", "", "")
    t.add_row("[bold yellow]Inference relevance[/]", "", "")
    t.add_row("  Tensor Cores", "4th-gen (FP8/FP16/BF16/INT8/INT4)",
              "10-100× speedup over CUDA cores for matmul")
    t.add_row("  Flash Attention", "Supported (sm_80+)",
              "Fused attention kernel — avoids materializing N×N matrix")
    t.add_row("  FP8 (Ada+)", "Yes" if cap >= "8.9" else "No",
              "Halves model size vs FP16 with ~equal quality")
    t.add_row("  Continuous batching", "Via vLLM / NIM",
              "Serve multiple users without padding waste")

    return Panel(t, title="[bold]NVIDIA GPU  ·  Inference Capabilities[/]",
                 border_style="green", padding=(0, 1))


def render_model_info(model_name: str, model_list: list[str]) -> Panel:
    """Panel showing available models and what they imply about VRAM."""
    lines = []
    for m in model_list[:8]:
        marker = "[bold green]▶[/]" if m == model_name else " "
        lines.append(f" {marker} {m}")
    body = "\n".join(lines) or "  (none — is the inference server running?)"
    return Panel(
        body + "\n\n[dim]VRAM rule of thumb:\n"
        "  FP16:  ~2 GB per 1 B params  (7B → ~14 GB)\n"
        "  INT8:  ~1 GB per 1 B params  (7B →  ~7 GB)\n"
        "  INT4:  ~0.5 GB per 1 B params (7B → ~3.5 GB)\n"
        "  L40S 48 GB can comfortably fit a 70B model in FP16.[/]",
        title="[bold]Available Models[/]",
        border_style="cyan",
        padding=(0, 1),
    )


def render_metrics_panel(m: InferenceMetrics, gpu: GPUMonitor,
                         turn: int, ctx_tokens: int) -> Panel:
    """Post-response metrics panel with educational annotations."""
    gpu_stats = gpu.stats()

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("Section", style="bold cyan", width=26)
    t.add_column("Value",   style="white")
    t.add_column("Insight", style="dim")

    # ── TOKENS ────────────────────────────────────────────────────────────────
    t.add_row("[bold yellow]TOKENS[/]", "", "")
    t.add_row("  Prompt (input)",    f"[green]{m.prompt_tokens:,}[/] tok",
              "Prefill: processed in parallel")
    t.add_row("  Generated (output)",f"[green]{m.output_tokens:,}[/] tok",
              "Decode: one token per forward pass")
    t.add_row("  Context so far",    f"[cyan]{ctx_tokens:,}[/] tok",
              "Accumulates in KV cache across turns")

    # ── LATENCY ───────────────────────────────────────────────────────────────
    t.add_row("", "", "")
    t.add_row("[bold yellow]LATENCY[/]", "", "")
    ttft_c = "green" if m.ttft < 0.5 else ("yellow" if m.ttft < 2.0 else "red")
    t.add_row("  Time to First Token",
              f"[{ttft_c}]{m.ttft:.3f}s[/]",
              "Prefill done → user sees first token")
    t.add_row("  Total wall time",
              f"{m.total_wall:.2f}s", "")

    # ── THROUGHPUT ────────────────────────────────────────────────────────────
    t.add_row("", "", "")
    t.add_row("[bold yellow]THROUGHPUT[/]", "", "")
    gen_c = "green" if m.generation_tps > 40 else (
            "yellow" if m.generation_tps > 15 else "red")
    t.add_row("  Decode speed",
              f"[{gen_c}]{m.generation_tps:.1f} tok/s[/]",
              "Memory-bandwidth bound — limited by VRAM BW")
    if m.prefill_tps > 0:
        t.add_row("  Prefill speed",
                  f"{m.prefill_tps:.1f} tok/s",
                  "Compute-bound — highly parallel, faster")

    # ── GPU (live) ─────────────────────────────────────────────────────────────
    if gpu_stats:
        t.add_row("", "", "")
        t.add_row("[bold yellow]GPU  (during generation)[/]", "", "")

        util_pct = gpu_stats["avg_util"]
        util_c   = _color_pct(util_pct)
        t.add_row("  SM Utilization",
                  f"[{util_c}]{util_pct:.0f}% avg  /  {gpu_stats['peak_util']:.0f}% peak[/]"
                  f"  [{util_c}]{_bar(util_pct, 16)}[/]",
                  "Low util during decode is normal — BW bound")

        mem_pct  = gpu_stats["avg_mem_gb"] / gpu_stats["total_mem_gb"] * 100
        mem_c    = _color_pct(mem_pct)
        t.add_row("  VRAM Used",
                  f"[{mem_c}]{gpu_stats['avg_mem_gb']:.1f} / "
                  f"{gpu_stats['total_mem_gb']:.0f} GB  ({mem_pct:.0f}%)[/]"
                  f"  [{mem_c}]{_bar(mem_pct, 16)}[/]",
                  "Weights + KV cache + activations")

        if gpu_stats["avg_temp_c"] > 0:
            temp_c_val = gpu_stats["avg_temp_c"]
            temp_color = _color_pct(temp_c_val, 70, 85)
            t.add_row("  Temperature",
                      f"[{temp_color}]{temp_c_val:.0f}°C avg  /  "
                      f"{gpu_stats['peak_temp_c']:.0f}°C peak[/]", "")

        if gpu_stats.get("avg_sm_clk", 0) > 0:
            t.add_row("  SM Clock",
                      f"{gpu_stats['avg_sm_clk']:,} MHz",
                      "May throttle under sustained load or thermal pressure")

    # ── WHAT JUST HAPPENED ────────────────────────────────────────────────────
    t.add_row("", "", "")
    t.add_row("[bold yellow]WHAT JUST HAPPENED[/]", "", "")
    t.add_row("  1. Tokenize",
              "text → token IDs",
              "BPE / SentencePiece vocab map")
    t.add_row("  2. Prefill",
              f"{m.prompt_tokens} tokens → KV cache",
              "Attention across all input tokens in parallel")
    t.add_row("  3. Decode loop",
              f"{m.output_tokens} × forward pass",
              "Each pass: matmul on weights + attend to KV cache")
    t.add_row("  4. Sample / argmax",
              "logits → next token",
              "Temperature, top-p, top-k applied here")
    t.add_row("  5. Detokenize",
              "token IDs → text",
              "Vocab lookup; streamed to your terminal")

    return Panel(t,
                 title=f"[bold]Turn {turn}  ·  Inference Metrics[/]",
                 border_style="bright_blue",
                 padding=(0, 1))


def render_help() -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("Command", style="bold cyan", width=18)
    t.add_column("Description", style="white")
    t.add_row("/help",      "Show this panel")
    t.add_row("/batch",     "★  Continuous batching demo — the NIM 'aha' moment")
    t.add_row("/bench",     "Run throughput benchmark (sequential requests)")
    t.add_row("/gpu",       "Print current GPU snapshot")
    t.add_row("/models",    "List available models on the server")
    t.add_row("/model <n>", "Switch to a different model")
    t.add_row("/clear",     "Reset conversation history")
    t.add_row("/quantize",  "Explain quantization (educational)")
    t.add_row("/kvexplain", "Explain the KV cache (educational)")
    t.add_row("/exit",      "Quit")
    return Panel(t, title="[bold]Commands[/]", border_style="cyan", padding=(0, 1))


# ── Educational explainers ────────────────────────────────────────────────────

QUANTIZE_EXPLAINER = """\
[bold yellow]Quantization — fitting bigger models into GPU memory[/]

Every LLM weight is a floating-point number. The format determines precision:

  [bold]FP32[/]   4 bytes/param   Original training format; rarely used for inference
  [bold]BF16[/]   2 bytes/param   Default for modern GPUs (same exponent range as FP32)
  [bold]FP16[/]   2 bytes/param   Slightly different range; Flash Attention uses this
  [bold]INT8[/]   1 byte/param    ≈2× memory saving; ~1-2% quality drop
  [bold]FP8[/]    1 byte/param    Ada Lovelace / Hopper native; vLLM + TRT-LLM support
  [bold]INT4[/]   0.5 bytes/param ≈4× saving; noticeable quality drop on reasoning tasks
  [bold]GPTQ/AWQ[/] ~4-bit        Learned quantization; much better quality than naive INT4

[bold]L40S example (48 GB VRAM):[/]
  Llama-3.1-70B in BF16 = 140 GB  → [red]doesn't fit[/]
  Llama-3.1-70B in FP8  =  70 GB  → [red]doesn't fit[/] (single GPU)
  Llama-3.1-70B in INT4  =  35 GB → [green]fits![/]  (or use 2× L40S in FP16)
  Llama-3.1-8B  in BF16 =  16 GB  → [green]fits easily[/]

[bold]How NVIDIA hardware accelerates quantized inference:[/]
  • Tensor Cores natively execute INT8 and FP8 matrix multiply
  • FP8 on Ada = same throughput as INT8, but higher quality
  • NVIDIA NIM and TensorRT-LLM auto-select the best precision
"""

KVCACHE_EXPLAINER = """\
[bold yellow]KV Cache — the memory that makes chat possible[/]

Transformer attention computes:
  attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V

During autoregressive decode, [bold]Q is just the new token[/], but K and V
span the entire conversation. Re-computing K and V from scratch each step
would cost O(sequence²) — prohibitively slow.

The KV cache stores each layer's K and V tensors so they only need to be
computed once per token as it enters the context.

[bold]VRAM cost:[/]
  KV cache ≈ 2 × num_layers × num_heads × head_dim × seq_len × dtype_bytes
  For Llama-3.1-8B (32 layers, 32 heads, 128 dim) in FP16:
    Per token = 2 × 32 × 32 × 128 × 2 = 524 KB
    For 8192-token context = ~4 GB  ← this is why context length matters!

[bold]Optimizations used on NVIDIA GPUs:[/]
  • [bold]Flash Attention[/]   — computes attention in chunks to fit L1/L2 cache
  • [bold]Paged Attention[/]   — vLLM allocates KV cache in pages (like OS virtual memory)
    Avoids fragmentation; enables continuous batching across users
  • [bold]Prefix caching[/]    — reuse KV cache for identical prompt prefixes
    (system prompts, few-shot examples saved for free)
  • [bold]Speculative decoding[/] — draft model generates candidates; verify in parallel
"""


# ── Continuous Batching Demo ──────────────────────────────────────────────────

# Short, deterministic prompt — same for every user so timing differences
# come purely from queuing, not content variance.
_BATCH_PROMPT = (
    "In exactly two sentences, explain what a GPU tensor core does "
    "and why it matters for AI inference."
)
_BATCH_MAX_TOKENS = 80
_BATCH_N          = 6   # number of simulated users


def _timed_request(client: openai.OpenAI, model: str,
                   prompt: str, max_tokens: int, user_idx: int) -> dict:
    """
    Send one streaming request and return a timing dict.
    Designed to be called from multiple threads simultaneously.
    """
    r: dict = {
        "user":          user_idx + 1,
        "t_sent":        None,
        "t_first_token": None,
        "t_done":        None,
        "tokens":        0,
        "error":         None,
    }
    try:
        r["t_sent"] = time.perf_counter()
        first = True
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=max_tokens,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if first:
                    r["t_first_token"] = time.perf_counter()
                    first = False
                r["tokens"] += 1
            if hasattr(chunk, "usage") and chunk.usage:
                if getattr(chunk.usage, "completion_tokens", 0):
                    r["tokens"] = chunk.usage.completion_tokens
        r["t_done"] = time.perf_counter()
    except Exception as e:
        r["error"] = str(e)
        r["t_done"] = time.perf_counter()
    return r


def _gantt_panel(seq_results: list[dict], con_results: list[dict],
                 seq_start: float, con_start: float) -> Panel:
    """
    Side-by-side ASCII Gantt chart — the visual 'aha moment'.

    Each row is one user. The bar spans from when their request was sent
    to when the last token arrived, normalised to the same total span so
    both charts are directly comparable.
    """
    CHART_W = 40   # characters wide

    def _row(t_sent: float, t_done: float, batch_start: float,
             total_span: float, color: str) -> str:
        if total_span <= 0:
            return ""
        start_frac = max(0.0, (t_sent - batch_start) / total_span)
        done_frac  = min(1.0, (t_done  - batch_start) / total_span)
        col_s = int(start_frac * CHART_W)
        col_e = max(col_s + 1, int(done_frac * CHART_W))
        bar   = " " * col_s + "█" * (col_e - col_s)
        secs  = t_done - t_sent
        return f"[{color}]{bar:<{CHART_W}}[/]  [dim]{secs:.1f}s[/]"

    # Use the longer of the two total spans as the common scale so bars
    # are directly comparable.
    seq_span = (seq_results[-1]["t_done"]  - seq_start)  if seq_results else 1
    con_span = (con_results[-1]["t_done"]  - con_start)  if con_results else 1
    total_span = max(seq_span, con_span)

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold",
              padding=(0, 1))
    t.add_column("User",    style="bold cyan",  width=6,  no_wrap=True)
    t.add_column(f"Sequential  (total {seq_span:.1f}s)",
                 style="white", width=CHART_W + 8, no_wrap=True)
    t.add_column(f"Concurrent  (total {con_span:.1f}s)",
                 style="white", width=CHART_W + 8, no_wrap=True)

    for i in range(max(len(seq_results), len(con_results))):
        sr = seq_results[i] if i < len(seq_results) else None
        cr = con_results[i] if i < len(con_results) else None

        seq_bar = _row(sr["t_sent"], sr["t_done"], seq_start,
                       total_span, "yellow") if sr else ""
        con_bar = _row(cr["t_sent"], cr["t_done"], con_start,
                       total_span, "bright_green") if cr else ""

        t.add_row(f"User {i + 1}", seq_bar, con_bar)

    speedup = seq_span / con_span if con_span > 0 else 1.0

    return Panel(
        t,
        title="[bold]Request Timeline  ·  same horizontal scale[/]",
        subtitle=(
            f"[bold bright_green]{speedup:.1f}× faster total wall time "
            f"with concurrent batching[/]"
            if speedup > 1.1 else
            "[dim]Run against vLLM / NIM for larger speedup via continuous batching[/]"
        ),
        border_style="bright_magenta",
        padding=(0, 1),
    )


def run_batching_demo(client: openai.OpenAI, model: str, gpu: GPUMonitor):
    """
    The continuous-batching 'aha' demo.

    Phase 1 — Sequential:
      Send _BATCH_N requests one at a time.  User N waits for users 1…N-1
      to finish before their request even starts.  GPU idles between requests.

    Phase 2 — Concurrent:
      Fire all _BATCH_N requests simultaneously from separate threads.
      With a NIM / vLLM backend the server uses continuous batching:
        • Prefill all sequences in one fused forward pass
        • Decode all sequences together — one GPU step produces one token
          for EVERY active user, not just one
      Total wall time ≈ a single request, regardless of user count.

    The Gantt chart at the end makes the difference unmissable.
    """
    n = _BATCH_N

    console.print()
    console.print(Rule("[bold bright_magenta]Continuous Batching Demo[/]",
                       style="bright_magenta"))
    console.print()
    console.print(Panel(
        "[bold yellow]What happens when multiple users ask at the same time?[/]\n\n"
        f"This demo sends [bold]{n} identical requests[/] in two modes:\n\n"
        "  [bold yellow]Phase 1[/]  Sequential  — "
        "each request waits for the previous to finish\n"
        "  [bold bright_green]Phase 2[/]  Concurrent  — "
        "all requests fire simultaneously\n\n"
        "[dim]With NIM / vLLM the server batches all users into the same GPU "
        "forward pass.\nWith Ollama the server queues them — same result as "
        "sequential.[/]",
        border_style="bright_magenta",
        padding=(0, 2),
    ))
    input("\n  Press [Enter] to start...\n")

    # ── Phase 1 : Sequential ──────────────────────────────────────────────────
    console.print(Rule("[yellow]Phase 1 — Sequential  (no batching)[/]",
                       style="yellow"))
    console.print(
        "[dim]Sending one request at a time — "
        "later users sit in line waiting.[/]\n"
    )

    seq_results: list[dict] = []
    seq_start = time.perf_counter()

    gpu.start()
    for i in range(n):
        console.print(
            f"  [dim]→ Sending request for User {i + 1} …[/]", end=" "
        )
        r = _timed_request(client, model, _BATCH_PROMPT,
                           _BATCH_MAX_TOKENS, i)
        r["t_sent_rel"] = r["t_sent"] - seq_start
        r["t_done_rel"] = r["t_done"] - seq_start
        seq_results.append(r)

        if r["error"]:
            console.print(f"[red]error: {r['error']}[/]")
        else:
            waited  = r["t_sent_rel"]
            elapsed = r["t_done"] - r["t_sent"]
            console.print(
                f"[yellow]waited {waited:.1f}s in queue[/]  →  "
                f"response in {elapsed:.1f}s  "
                f"([dim]{r['tokens']} tok[/])  [green]✓[/]"
            )
    gpu.stop()

    seq_total = seq_results[-1]["t_done_rel"] if seq_results else 0
    seq_gpu   = gpu.stats()

    console.print(
        f"\n  [yellow]Total wall time: {seq_total:.1f}s[/]  "
        f"([dim]GPU avg util: "
        f"{seq_gpu['avg_util']:.0f}%[/])\n"
        if seq_gpu else
        f"\n  [yellow]Total wall time: {seq_total:.1f}s[/]\n"
    )

    # ── Explain what just happened ────────────────────────────────────────────
    console.print(Panel(
        "[bold yellow]What you just saw (sequential):[/]\n\n"
        f"  User 1 got their answer right away.\n"
        f"  User {n} had to wait ~{seq_results[-2]['t_done_rel']:.0f}s "
        f"before the server even looked at their message.\n\n"
        "  The GPU was [bold]idle between each request[/] — "
        "it finished one user's tokens,\n"
        "  took a breath, then started the next. "
        "Wasted capacity every time.\n\n"
        "[dim]This is how a naive single-threaded server works.[/]",
        border_style="yellow",
        padding=(0, 2),
    ))
    input("  Press [Enter] to run Phase 2...\n")

    # ── Phase 2 : Concurrent ──────────────────────────────────────────────────
    console.print(Rule("[bright_green]Phase 2 — Concurrent  (continuous batching)[/]",
                       style="bright_green"))
    console.print(
        f"[dim]Firing all {n} requests simultaneously — "
        "watch how long User {n} waits now.[/]\n"
    )

    con_results_raw: dict[int, dict] = {}
    con_start = time.perf_counter()
    lock = threading.Lock()

    # Track live completions
    completed = [0]

    def _worker(idx: int):
        r = _timed_request(client, model, _BATCH_PROMPT,
                           _BATCH_MAX_TOKENS, idx)
        with lock:
            con_results_raw[idx] = r
            completed[0] += 1
            elapsed = r["t_done"] - r["t_sent"]
            waited  = r["t_sent"] - con_start
            status  = "[red]✗[/]" if r["error"] else "[green]✓[/]"
            console.print(
                f"  User {idx + 1} done  "
                f"[dim]sent +{waited:.2f}s[/]  →  "
                f"response in {elapsed:.1f}s  "
                f"([dim]{r['tokens']} tok[/])  {status}"
            )

    gpu.start()
    console.print(f"  [bright_green]→ Sending all {n} requests NOW...[/]\n")
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_worker, i) for i in range(n)]
        for f in as_completed(futures):
            f.result()  # surface any exceptions
    gpu.stop()

    con_results = [con_results_raw[i] for i in range(n)
                   if i in con_results_raw]
    con_total = (max(r["t_done"] for r in con_results) - con_start) \
                if con_results else 0
    con_gpu = gpu.stats()

    console.print(
        f"\n  [bright_green]Total wall time: {con_total:.1f}s[/]  "
        f"([dim]GPU avg util: "
        f"{con_gpu['avg_util']:.0f}%[/])\n"
        if con_gpu else
        f"\n  [bright_green]Total wall time: {con_total:.1f}s[/]\n"
    )

    # ── Gantt comparison ──────────────────────────────────────────────────────
    console.print(Rule("[bold]Timeline Comparison[/]", style="bright_magenta"))
    console.print()
    console.print(_gantt_panel(seq_results, con_results, seq_start, con_start))

    # ── The explanation ───────────────────────────────────────────────────────
    speedup = seq_total / con_total if con_total > 0 else 1.0
    gpu_diff = ""
    if seq_gpu and con_gpu:
        gpu_diff = (
            f"\n\n  GPU utilization:\n"
            f"    Sequential   {seq_gpu['avg_util']:.0f}% avg — "
            f"idles between each user's tokens\n"
            f"    Concurrent   {con_gpu['avg_util']:.0f}% avg — "
            f"kept busy serving all users at once"
        )

    console.print(Panel(
        f"[bold bright_green]What NIM / vLLM does differently:[/]\n\n"
        "  With [bold]continuous batching[/], the server doesn't wait for a "
        "request to finish\n"
        "  before accepting the next one. Instead:\n\n"
        "    1. Every incoming request joins the [bold]active batch[/] immediately\n"
        "    2. Each GPU forward pass produces one token for [bold]every active "
        "user[/]\n"
        "    3. Finished sequences are swapped out; new ones slip in — "
        "[bold]no idle time[/]\n\n"
        f"  Result: {n} users got answers in [bold bright_green]{con_total:.1f}s[/] "
        f"instead of [bold yellow]{seq_total:.1f}s[/]  "
        f"([bold bright_green]{speedup:.1f}× faster[/] total throughput)"
        f"{gpu_diff}\n\n"
        "[dim]The GPU does the same amount of work either way — batching just "
        "stops it from waiting around between users.[/]",
        title="[bold]Why Continuous Batching Matters[/]",
        border_style="bright_green",
        padding=(1, 2),
    ))


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(client: openai.OpenAI, model: str, gpu: GPUMonitor,
                  n: int = 8):
    """
    Send n identical short prompts in sequence, measure aggregate tok/s.
    Demonstrates how the server amortises batching overhead.
    """
    console.print()
    console.print(Rule("[bold]Throughput Benchmark[/]", style="yellow"))
    console.print(f"[dim]Sending {n} requests to [cyan]{model}[/] ...[/]\n")

    prompt = "List 5 interesting facts about GPU architecture in a numbered list."
    results: list[InferenceMetrics] = []

    gpu.start()
    for i in range(1, n + 1):
        m  = InferenceMetrics()
        m.t_request = time.perf_counter()
        buf = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=256,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    if not m.first_token_received:
                        m.t_first_token = time.perf_counter()
                        m.first_token_received = True
                    buf += delta.content
                    m.output_tokens += 1
                if chunk.choices[0].finish_reason:
                    m.finish_reason = chunk.choices[0].finish_reason
                if hasattr(chunk, "usage") and chunk.usage:
                    m.prompt_tokens  = chunk.usage.prompt_tokens
                    m.output_tokens  = chunk.usage.completion_tokens
            m.t_done = time.perf_counter()
        except Exception as e:
            console.print(f"[red]Request {i} failed: {e}[/]")
            continue

        results.append(m)
        tps = m.generation_tps
        c = "green" if tps > 40 else ("yellow" if tps > 15 else "red")
        console.print(f"  [{i:2d}/{n}]  TTFT {m.ttft:.3f}s  "
                      f"decode [{c}]{tps:.1f} tok/s[/]  "
                      f"output {m.output_tokens} tok")

    gpu.stop()

    if not results:
        console.print("[red]No successful requests.[/]")
        return

    avg_tps   = sum(r.generation_tps for r in results) / len(results)
    avg_ttft  = sum(r.ttft            for r in results) / len(results)
    total_tok = sum(r.output_tokens   for r in results)
    wall      = results[-1].t_done - results[0].t_request
    agg_tps   = total_tok / wall if wall > 0 else 0

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    t.add_column("Metric",   style="bold")
    t.add_column("Value",    style="white")
    t.add_column("Notes",    style="dim")
    t.add_row("Requests",           str(len(results)), "sequential")
    t.add_row("Avg TTFT",           f"{avg_ttft:.3f}s",
              "prefill latency per request")
    t.add_row("Avg decode speed",   f"{avg_tps:.1f} tok/s",
              "per-request throughput")
    t.add_row("Aggregate tok/s",    f"{agg_tps:.1f} tok/s",
              "total output / total wall time")
    t.add_row("Total tokens out",   f"{total_tok:,}", "")

    gs = gpu.stats()
    if gs:
        t.add_row("GPU util (avg)",  f"{gs['avg_util']:.0f}%", "")
        t.add_row("VRAM peak",       f"{gs['peak_mem_gb']:.1f} GB", "")

    console.print()
    console.print(Panel(t, title="[bold]Benchmark Results[/]", border_style="yellow"))
    console.print(
        "[dim]\nTip: Run [bold]/batch[/] to see a side-by-side sequential vs "
        "concurrent demo — the visual that makes continuous batching click.[/]"
    )


# ── Server connectivity ───────────────────────────────────────────────────────

def detect_models(client: openai.OpenAI) -> list[str]:
    """Return sorted list of model IDs from the inference server."""
    try:
        return sorted(m.id for m in client.models.list().data)
    except Exception:
        return []


def pick_model(models: list[str], hint: str) -> str:
    """Select a model: prefer hint, then first available, else ask."""
    if hint and hint in models:
        return hint
    if hint:
        # partial match
        for m in models:
            if hint.lower() in m.lower():
                return m
    if models:
        return models[0]
    return hint or "default"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.print()
    console.print(Panel(
        "[bold white]NVIDIA Inference Demo[/]\n"
        "[dim]Interactive LLM chat with real-time GPU metrics and educational annotations.\n"
        "Powered by any OpenAI-compatible inference server "
        "(vLLM · NVIDIA NIM · TensorRT-LLM).[/]",
        border_style="bright_green",
        padding=(0, 2),
    ))

    # ── GPU setup ─────────────────────────────────────────────────────────────
    console.print("[dim]Probing GPU...[/]", end=" ")
    gpu = GPUMonitor()
    if gpu.available:
        console.print(f"[green]{gpu.gpu_name}  ({gpu.total_mem_gb:.0f} GB)[/]")
    else:
        console.print("[yellow]No NVIDIA GPU found — GPU metrics will be skipped[/]")

    console.print(render_gpu_header(gpu))

    # ── Connect to inference server ───────────────────────────────────────────
    console.print(f"[dim]Connecting to inference server at [cyan]{BASE_URL}[/]...[/]",
                  end=" ")
    client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)
    models = detect_models(client)

    if not models:
        console.print()
        console.print(Panel(
            f"[yellow]Could not reach inference server at [bold]{BASE_URL}[/][/]\n\n"
            "[bold]Start one of:[/]\n\n"
            "  [bold cyan]vLLM[/] (most common):\n"
            "    pip install vllm\n"
            "    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000\n\n"
            "  [bold cyan]NVIDIA NIM[/] (Docker, requires NVIDIA NGC account):\n"
            "    docker run --gpus all -p 8000:8000 \\\n"
            "      nvcr.io/nim/meta/llama-3.1-8b-instruct:latest\n\n"
            "  [bold cyan]Ollama (OpenAI compat)[/]:\n"
            "    ollama serve\n"
            "    export INFERENCE_BASE_URL=http://localhost:11434/v1\n\n"
            "[dim]Then re-run this script.[/]",
            border_style="yellow",
            title="[bold]Server Not Found[/]",
            padding=(1, 2),
        ))
        sys.exit(1)

    console.print(f"[green]connected.[/]  {len(models)} model(s) available.")

    model = pick_model(models, MODEL_HINT)
    console.print(render_model_info(model, models))
    console.print(f"[bold green]Active model:[/] [cyan]{model}[/]")
    console.print()
    console.print(
        "[dim]Type a message to chat, or [bold]/help[/] for commands.\n"
        "  ★  Try [bold bright_magenta]/batch[/] to see the continuous batching "
        "demo — the clearest way to understand what NIM does differently.[/]"
    )
    console.print()

    # ── Chat state ────────────────────────────────────────────────────────────
    history: list[dict] = []
    ctx_tokens: int = 0
    turn: int = 0

    while True:
        try:
            user_input = Prompt.ask("[bold bright_green]You[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/]")
            break

        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────────────────

        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()

            if cmd in ("/exit", "/quit", "/q"):
                console.print("[dim]Goodbye.[/]")
                break

            elif cmd == "/help":
                console.print(render_help())
                continue

            elif cmd == "/clear":
                history.clear()
                ctx_tokens = 0
                turn = 0
                console.print("[dim]Conversation cleared.[/]")
                continue

            elif cmd == "/models":
                models = detect_models(client)
                console.print(render_model_info(model, models))
                continue

            elif cmd == "/model":
                if len(cmd_parts) < 2:
                    console.print("[yellow]Usage: /model <name>[/]")
                    continue
                new_model = cmd_parts[1].strip()
                all_models = detect_models(client)
                picked = pick_model(all_models, new_model)
                model = picked
                console.print(f"[green]Switched to:[/] [cyan]{model}[/]")
                history.clear()
                ctx_tokens = 0
                turn = 0
                continue

            elif cmd == "/gpu":
                snap = gpu.snapshot()
                if snap:
                    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
                    t.add_column("K", style="bold cyan", width=20)
                    t.add_column("V", style="white")
                    util_c = _color_pct(snap["util"])
                    mem_pct = snap["mem_gb"] / snap["total_gb"] * 100
                    mem_c = _color_pct(mem_pct)
                    t.add_row("SM Utilization",
                              f"[{util_c}]{snap['util']:.0f}%[/]  "
                              f"[{util_c}]{_bar(snap['util'], 20)}[/]")
                    t.add_row("VRAM",
                              f"[{mem_c}]{snap['mem_gb']:.1f} / "
                              f"{snap['total_gb']:.0f} GB  ({mem_pct:.0f}%)[/]  "
                              f"[{mem_c}]{_bar(mem_pct, 20)}[/]")
                    if snap.get("temp_c", 0):
                        tc = _color_pct(snap["temp_c"], 70, 85)
                        t.add_row("Temperature",
                                  f"[{tc}]{snap['temp_c']:.0f}°C[/]")
                    if snap.get("sm_clk", 0):
                        t.add_row("SM Clock",
                                  f"{snap['sm_clk']:,} MHz")
                    console.print(Panel(t, title="[bold]GPU Snapshot[/]",
                                        border_style="green"))
                else:
                    console.print("[yellow]GPU monitoring not available.[/]")
                continue

            elif cmd == "/batch":
                run_batching_demo(client, model, gpu)
                continue

            elif cmd == "/bench":
                run_benchmark(client, model, gpu)
                continue

            elif cmd == "/quantize":
                console.print(Panel(QUANTIZE_EXPLAINER,
                                    title="[bold]Quantization[/]",
                                    border_style="yellow", padding=(1, 2)))
                continue

            elif cmd == "/kvexplain":
                console.print(Panel(KVCACHE_EXPLAINER,
                                    title="[bold]KV Cache[/]",
                                    border_style="yellow", padding=(1, 2)))
                continue

            else:
                console.print(f"[yellow]Unknown command: {cmd}  (try /help)[/]")
                continue

        # ── Chat turn ─────────────────────────────────────────────────────────

        turn += 1
        history.append({"role": "user", "content": user_input})

        m = InferenceMetrics()
        m.t_request = time.perf_counter()

        gpu.start()

        response_text = ""
        console.print()
        console.print(f"[bold bright_blue]Assistant[/] [dim]({model})[/]")

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=history,
                stream=True,
                stream_options={"include_usage": True},
            )

            for chunk in stream:
                if not chunk.choices:
                    # usage-only chunk (OpenAI stream_options)
                    if hasattr(chunk, "usage") and chunk.usage:
                        m.prompt_tokens  = chunk.usage.prompt_tokens
                        m.output_tokens  = chunk.usage.completion_tokens
                    continue

                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if not m.first_token_received:
                        m.t_first_token = time.perf_counter()
                        m.first_token_received = True
                    response_text += delta.content
                    console.print(delta.content, end="", markup=False)

                if chunk.choices[0].finish_reason:
                    m.finish_reason = chunk.choices[0].finish_reason

                if hasattr(chunk, "usage") and chunk.usage:
                    m.prompt_tokens  = chunk.usage.prompt_tokens or m.prompt_tokens
                    m.output_tokens  = chunk.usage.completion_tokens or m.output_tokens

        except openai.APIConnectionError:
            console.print(
                f"\n[red]Connection lost to {BASE_URL}. "
                "Is the inference server still running?[/]"
            )
            gpu.stop()
            history.pop()
            continue
        except openai.APIError as e:
            console.print(f"\n[red]API error: {e}[/]")
            gpu.stop()
            history.pop()
            continue
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/]")
            gpu.stop()
            history.pop()
            continue

        m.t_done = time.perf_counter()
        gpu.stop()

        # If the server didn't return usage, estimate from token count
        if m.output_tokens == 0:
            m.output_tokens = len(response_text.split())  # rough estimate

        console.print()
        console.print()

        # Accumulate context tokens
        if m.prompt_tokens:
            ctx_tokens = m.prompt_tokens + m.output_tokens
        else:
            ctx_tokens += m.output_tokens + len(user_input.split())

        console.print(render_metrics_panel(m, gpu, turn, ctx_tokens))

        history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
