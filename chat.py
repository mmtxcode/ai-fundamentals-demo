#!/usr/bin/env python3
"""
AI Fundamentals Demo — Interactive CLI Chat with Performance Metrics
====================================================================
Educational tool for understanding how AI models work under the hood.
Powered by Ollama for local model inference.

Run:  ./run.sh          (macOS / Linux — sets up venv automatically)
      run.bat           (Windows)
      python chat.py    (if dependencies are already installed)
"""

import sys
import os
import asyncio
import platform
import subprocess
import time
import threading
import socket

import ollama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.prompt import Prompt
from rich.rule import Rule
from rich.columns import Columns
from rich.markup import escape

console = Console()

# ── Ollama lifecycle ──────────────────────────────────────────────────────────

def _ollama_reachable() -> bool:
    """Check if Ollama's HTTP API is up."""
    try:
        s = socket.create_connection(("127.0.0.1", 11434), timeout=1)
        s.close()
        return True
    except OSError:
        return False


def _ollama_in_path() -> bool:
    cmd = "where" if platform.system() == "Windows" else "which"
    return subprocess.run([cmd, "ollama"], capture_output=True).returncode == 0


def _install_ollama() -> bool:
    """
    Attempt to install Ollama for the current OS.
    Returns True if installation succeeded.
    """
    system = platform.system()
    console.print()

    if system == "Darwin":
        # Prefer Homebrew — no curl | sh required
        if subprocess.run(["which", "brew"], capture_output=True).returncode == 0:
            console.print("[dim]Installing Ollama via Homebrew...[/]")
            result = subprocess.run(["brew", "install", "ollama"])
            return result.returncode == 0
        else:
            console.print(
                "[yellow]Homebrew not found.[/] Install Ollama manually:\n"
                "  1. Download the macOS app from [bold]https://ollama.com/download[/]\n"
                "  2. Or install Homebrew first: [bold]https://brew.sh[/]"
            )
            return False

    elif system == "Linux":
        console.print(
            "This will run the official Ollama install script:\n"
            "  [bold]curl -fsSL https://ollama.com/install.sh | sh[/]"
        )
        confirm = Prompt.ask("Proceed?", choices=["y", "n"], default="y")
        if confirm != "y":
            return False
        result = subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True,
        )
        return result.returncode == 0

    else:  # Windows
        console.print(
            "[yellow]Automatic install not supported on Windows.[/]\n"
            "Download the installer from [bold]https://ollama.com/download[/]"
        )
        return False


def ensure_ollama():
    """Install Ollama if missing, then start it if not already running."""
    if _ollama_reachable():
        return

    if not _ollama_in_path():
        console.print(Panel(
            "[yellow]Ollama is not installed.[/]\n\n"
            "Ollama runs AI models locally on your machine.\n"
            "Would you like to install it now?",
            border_style="yellow",
            padding=(0, 2),
        ))
        choice = Prompt.ask("Install Ollama?", choices=["y", "n"], default="y")
        if choice != "y":
            console.print("[red]Ollama is required to run this demo.[/]")
            sys.exit(1)

        if not _install_ollama():
            console.print("[red]Installation failed or was skipped.[/] Please install Ollama manually.")
            sys.exit(1)

        # Reload PATH so the newly installed binary is found
        import os
        os.environ["PATH"] += os.pathsep + "/usr/local/bin"

        if not _ollama_in_path():
            console.print(
                "[yellow]Ollama installed but not found in PATH.[/]\n"
                "You may need to open a new terminal, then re-run this script."
            )
            sys.exit(1)

    console.print("[dim]Starting Ollama...[/]", end=" ")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(20):
        time.sleep(0.5)
        if _ollama_reachable():
            console.print("[green]ready.[/]")
            return

    console.print("[red]Timed out waiting for Ollama to start.[/]")
    sys.exit(1)


# ── MCP Client ───────────────────────────────────────────────────────────────

class MCPClient:
    """
    Connects to the Intersight MCP server process and bridges its tools
    to Ollama's function-calling format.

    Why this matters for the demo:
      Without tools the model answers from training data — it may hallucinate
      server counts, firmware versions, or alarm states. With tools it calls
      the live Intersight API and grounds its answer in real infrastructure data.
      The difference is visible: tool calls appear inline before the response.
    """

    MCP_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intersight_mcp.py")

    def __init__(self):
        self.tools: list[dict] = []   # Ollama function-calling format
        self.available = False
        self.error = ""

    def setup(self) -> bool:
        """Spawn the MCP server, fetch its tool list, convert to Ollama format."""
        try:
            self.tools = asyncio.run(self._fetch_tools())
            self.available = bool(self.tools)
            return self.available
        except Exception as e:
            self.error = str(e)
            self.available = False
            return False

    def call(self, name: str, arguments: dict) -> str:
        """Call a single MCP tool and return its text result."""
        try:
            return asyncio.run(self._call_tool(name, arguments))
        except Exception as e:
            return f"Tool error: {e}"

    # ── Async internals ───────────────────────────────────────────────────────

    async def _fetch_tools(self) -> list[dict]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=[self.MCP_SERVER],
            env={**os.environ},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": t.inputSchema or {
                                "type": "object", "properties": {}
                            },
                        },
                    }
                    for t in result.tools
                ]

    async def _call_tool(self, name: str, arguments: dict) -> str:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=[self.MCP_SERVER],
            env={**os.environ},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return "\n".join(
                    c.text for c in result.content if hasattr(c, "text")
                )


# ── GPU Monitoring ────────────────────────────────────────────────────────────

def _cuda_arch_name(major: int, minor: int) -> str:
    """Map CUDA compute capability to marketing architecture name."""
    mapping = {
        (9, 0): "Hopper",
        (8, 9): "Ada Lovelace",
        (8, 6): "Ampere",
        (8, 0): "Ampere",
        (7, 5): "Turing",
        (7, 0): "Volta",
        (6, 1): "Pascal",
        (6, 0): "Pascal",
        (5, 2): "Maxwell",
        (5, 0): "Maxwell",
        (3, 7): "Kepler",
        (3, 5): "Kepler",
    }
    return mapping.get((major, minor), f"sm_{major}{minor}")


class GPUMonitor:
    """
    Samples GPU utilization and memory in a background thread during generation.

    Why this matters for AI infrastructure:
      GPU utilization tells you if the model is compute-bound or memory-bound.
      LLM inference is almost always memory-bandwidth bound — the GPU spends
      most cycles waiting on data, not computing. High util % but low throughput
      usually means you're bottlenecked on VRAM bandwidth, not FLOPS.
    """

    def __init__(self):
        self.samples_util: list[float] = []
        self.samples_mem_gb: list[float] = []
        self.total_mem_gb: float = 0.0
        self.running = False
        self._thread: threading.Thread | None = None
        self.available = False
        self.backend = None
        self._handle = None
        # Static hardware specs populated once at setup
        self.specs: dict = {}
        self._setup()

    def _setup(self):
        # Try pynvml first (NVIDIA) — richest data source
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            name  = pynvml.nvmlDeviceGetName(self._handle)
            self.gpu_name    = name if isinstance(name, str) else name.decode()
            self.total_mem_gb = info.total / 1024**3
            self.available   = True
            self.backend     = "pynvml"
            self.specs       = self._specs_pynvml()
            return
        except Exception:
            pass

        # Fall back to nvidia-smi subprocess
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
                self.specs        = self._specs_nvidia_smi(parts)
                return
        except Exception:
            pass

        # Apple Silicon — no GPU util, but we can surface chip info
        if platform.system() == "Darwin" and platform.processor() == "arm":
            self.specs    = self._specs_apple_silicon()
            self.gpu_name = self.specs.get("chip", "Apple Silicon")
            self.available = False   # no dynamic util sampling
            self.backend  = "apple"
            return

        self.available = False
        self.gpu_name  = "N/A"

    # ── Static spec collectors ────────────────────────────────────────────────

    def _specs_pynvml(self) -> dict:
        """Gather static NVIDIA specs via pynvml."""
        specs: dict = {}
        try:
            import pynvml
            # Compute capability → architecture name
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(self._handle)
            specs["compute_cap"] = f"{major}.{minor}"
            specs["architecture"] = _cuda_arch_name(major, minor)

            # Memory bandwidth: clock (MHz) × bus width (bits) × 2 (DDR) / 8 → GB/s
            mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
                self._handle, pynvml.NVML_CLOCK_MEM
            )
            bus_width_bits = pynvml.nvmlDeviceGetMemoryBusWidth(self._handle)
            specs["mem_bw_gbs"]   = round(mem_clock_mhz * 2 * bus_width_bits / 8 / 1000, 1)
            specs["mem_clock_mhz"] = mem_clock_mhz
            specs["bus_width"]    = bus_width_bits

            # CUDA / driver versions
            driver = pynvml.nvmlSystemGetDriverVersion()
            specs["driver"] = driver if isinstance(driver, str) else driver.decode()

            # SM (shader multiprocessor) count
            specs["sm_count"] = pynvml.nvmlDeviceGetNumGpuCores(self._handle)
        except Exception:
            pass
        return specs

    def _specs_nvidia_smi(self, csv_parts: list[str]) -> dict:
        """Gather static NVIDIA specs via nvidia-smi (fallback)."""
        specs: dict = {}
        try:
            if len(csv_parts) >= 3:
                specs["driver"] = csv_parts[2].strip()
            if len(csv_parts) >= 4:
                cap = csv_parts[3].strip()          # e.g. "8.6"
                specs["compute_cap"] = cap
                try:
                    major = int(cap.split(".")[0])
                    minor = int(cap.split(".")[1])
                    specs["architecture"] = _cuda_arch_name(major, minor)
                except Exception:
                    pass
            # Bandwidth not directly in this query; try a second call
            r2 = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.bandwidth",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if r2.returncode == 0 and r2.stdout.strip():
                specs["mem_bw_gbs"] = float(r2.stdout.strip())
        except Exception:
            pass
        return specs

    def _specs_apple_silicon(self) -> dict:
        """Surface Apple Silicon chip info via system_profiler."""
        specs: dict = {}
        try:
            import json as _json
            r = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True, text=True, timeout=5
            )
            data = _json.loads(r.stdout)
            hw   = data.get("SPHardwareDataType", [{}])[0]
            chip = hw.get("chip_type") or hw.get("cpu_type", "Apple Silicon")
            cores_cpu = hw.get("number_processors", "")
            mem_gb    = hw.get("physical_memory", "")
            specs["chip"]      = chip
            specs["cpu_cores"] = str(cores_cpu)
            specs["ram"]       = mem_gb          # unified memory

            # GPU core count from SPDisplaysDataType
            r2 = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=5
            )
            disp = _json.loads(r2.stdout).get("SPDisplaysDataType", [{}])[0]
            gpu_cores = disp.get("sppci_cores", "")
            if gpu_cores:
                specs["gpu_cores"] = gpu_cores
        except Exception:
            pass
        return specs

    def _sample(self):
        if self.backend == "pynvml":
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(util.gpu), info.used / 1024**3
        elif self.backend == "nvidia-smi":
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            parts = r.stdout.strip().split(", ")
            return float(parts[0]), float(parts[1]) / 1024
        return 0.0, 0.0

    def start(self):
        self.samples_util.clear()
        self.samples_mem_gb.clear()
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
                util, mem = self._sample()
                self.samples_util.append(util)
                self.samples_mem_gb.append(mem)
            except Exception:
                pass
            time.sleep(0.4)

    def stats(self) -> dict | None:
        if not self.available or not self.samples_util:
            return None
        return {
            "name": self.gpu_name,
            "avg_util": sum(self.samples_util) / len(self.samples_util),
            "peak_util": max(self.samples_util),
            "avg_mem_gb": sum(self.samples_mem_gb) / len(self.samples_mem_gb),
            "peak_mem_gb": max(self.samples_mem_gb),
            "total_mem_gb": self.total_mem_gb,
        }


# ── Metrics collection ────────────────────────────────────────────────────────

class InferenceMetrics:
    """Collected during a single chat turn."""
    def __init__(self):
        self.t_request: float = 0.0       # wall time when request sent
        self.t_first_token: float = 0.0   # wall time of first token received
        self.t_done: float = 0.0          # wall time when stream finished
        self.first_token_received = False

        # From Ollama final chunk
        self.prompt_tokens: int = 0       # tokens actually evaluated from prompt
        self.output_tokens: int = 0       # tokens generated
        self.prompt_eval_ns: int = 0      # time ollama spent on prompt (ns)
        self.eval_ns: int = 0             # time ollama spent generating (ns)
        self.total_ns: int = 0            # total time inside ollama (ns)
        self.load_ns: int = 0             # model load time (ns)

    # ── Derived metrics ───────────────────────────────────────────────────────

    @property
    def ttft(self) -> float:
        """Time to First Token (seconds) — latency before streaming starts."""
        if self.t_first_token and self.t_request:
            return self.t_first_token - self.t_request
        return 0.0

    @property
    def total_wall(self) -> float:
        """Total elapsed wall time (seconds)."""
        return self.t_done - self.t_request if self.t_done else 0.0

    @property
    def generation_tps(self) -> float:
        """Output tokens per second — the core throughput metric."""
        secs = self.eval_ns / 1e9
        return self.output_tokens / secs if secs > 0 else 0.0

    @property
    def prompt_tps(self) -> float:
        """Prompt processing speed (tokens/sec) — prefill throughput."""
        secs = self.prompt_eval_ns / 1e9
        return self.prompt_tokens / secs if secs > 0 else 0.0


# ── Rendering helpers ─────────────────────────────────────────────────────────

def render_metrics_panel(
    m: InferenceMetrics,
    num_ctx: int,
    total_ctx_tokens: int,
    gpu: GPUMonitor,
    turn: int,
    cached_tokens: int = 0,
) -> Panel:
    """
    Builds the metrics panel shown after each AI response.
    Each section maps to a core AI infrastructure concept.
    """
    gpu_stats = gpu.stats()

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Section", style="bold cyan", width=26)
    table.add_column("Value", style="white")
    table.add_column("Insight", style="dim")

    # ── TOKENS ────────────────────────────────────────────────────────────────
    table.add_row(
        "[bold yellow]TOKENS[/]", "", ""
    )
    table.add_row(
        "  Input (prompt)",
        f"[green]{m.prompt_tokens:,}[/] tok",
        "Tokens sent to the model for context"
    )
    table.add_row(
        "  Output (generated)",
        f"[green]{m.output_tokens:,}[/] tok",
        "Tokens the model produced"
    )
    total_tok = m.prompt_tokens + m.output_tokens
    table.add_row(
        "  Total this turn",
        f"[green]{total_tok:,}[/] tok",
        ""
    )

    # ── LATENCY ───────────────────────────────────────────────────────────────
    table.add_row("", "", "")
    table.add_row("[bold yellow]LATENCY[/]", "", "")
    ttft_color = "green" if m.ttft < 0.5 else ("yellow" if m.ttft < 2.0 else "red")
    table.add_row(
        "  Time to First Token",
        f"[{ttft_color}]{m.ttft:.3f}s[/]",
        "Prefill done → user sees first word"
    )
    table.add_row(
        "  Total generation",
        f"{m.total_wall:.2f}s",
        "Wall time for complete response"
    )
    if m.load_ns > 0:
        table.add_row(
            "  Model load",
            f"{m.load_ns / 1e9:.2f}s",
            "One-time cost to load weights to GPU"
        )

    # ── THROUGHPUT ────────────────────────────────────────────────────────────
    table.add_row("", "", "")
    table.add_row("[bold yellow]THROUGHPUT[/]", "", "")
    gen_color = "green" if m.generation_tps > 20 else ("yellow" if m.generation_tps > 8 else "red")
    table.add_row(
        "  Generation speed",
        f"[{gen_color}]{m.generation_tps:.1f} tok/s[/]",
        "Autoregressive decode — sequential, hard to parallelize"
    )
    if m.prompt_tps > 0:
        table.add_row(
            "  Prefill speed",
            f"{m.prompt_tps:.1f} tok/s",
            "Prompt processing — parallel, much faster"
        )

    # ── KV CACHE ──────────────────────────────────────────────────────────────
    table.add_row("", "", "")
    table.add_row("[bold yellow]KV CACHE (Context)[/]", "", "")
    ctx_pct = (total_ctx_tokens / num_ctx * 100) if num_ctx > 0 else 0
    bar_filled = int(ctx_pct / 5)  # 20 chars total
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    ctx_color = "green" if ctx_pct < 50 else ("yellow" if ctx_pct < 80 else "red")
    table.add_row(
        "  Context window used",
        f"[{ctx_color}]{total_ctx_tokens:,} / {num_ctx:,} ({ctx_pct:.1f}%)[/]",
        "Each token stored as a K+V vector in VRAM"
    )
    table.add_row(
        "  Usage bar",
        f"[{ctx_color}]{bar}[/]",
        ""
    )
    if cached_tokens > 0:
        table.add_row(
            "  Cache reuse",
            f"[cyan]{cached_tokens:,} tok[/] skipped",
            "Prefix matched — attention recompute avoided"
        )

    # Estimate KV cache memory footprint (rough, model-dependent)
    # Rule of thumb: ~2 * num_layers * num_heads * head_dim * 2 bytes * seq_len
    # Simplified: assume ~0.5 MB per 1k tokens for a 7B model (varies widely)
    kv_est_mb = total_ctx_tokens * 0.5  # rough estimate in MB
    table.add_row(
        "  Est. KV cache size",
        f"~{kv_est_mb:.0f} MB",
        "Rough estimate — scales linearly with context length"
    )

    # ── GPU ───────────────────────────────────────────────────────────────────
    table.add_row("", "", "")
    table.add_row("[bold yellow]HARDWARE[/]", "", "")
    specs = gpu.specs
    if gpu_stats:
        util_color = "green" if gpu_stats["avg_util"] > 50 else "yellow"
        mem_pct = gpu_stats["peak_mem_gb"] / gpu_stats["total_mem_gb"] * 100 if gpu_stats["total_mem_gb"] > 0 else 0
        mem_color = "green" if mem_pct < 70 else ("yellow" if mem_pct < 90 else "red")

        # ── Static specs ──────────────────────────────────────────────────────
        table.add_row(f"  [bold]{gpu_stats['name']}[/]", "", "")
        if specs.get("architecture"):
            table.add_row(
                "  Architecture",
                f"{specs['architecture']}  [dim](sm {specs.get('compute_cap', '')})[/]",
                "GPU microarchitecture generation"
            )
        if specs.get("mem_bw_gbs"):
            table.add_row(
                "  Memory bandwidth",
                f"[bold]{specs['mem_bw_gbs']:.0f} GB/s[/]",
                "Key bottleneck for LLM decode throughput"
            )
        if specs.get("bus_width"):
            table.add_row(
                "  Memory bus width",
                f"{specs['bus_width']} bit",
                "Wider bus → more data per clock cycle"
            )
        table.add_row(
            "  VRAM",
            f"{gpu_stats['total_mem_gb']:.1f} GB",
            "Sets the ceiling on model size + context length"
        )
        if specs.get("driver"):
            table.add_row("  Driver", specs["driver"], "")

        # ── Dynamic stats ─────────────────────────────────────────────────────
        table.add_row("", "", "")
        table.add_row(
            "  GPU utilization",
            f"[{util_color}]avg {gpu_stats['avg_util']:.0f}%  peak {gpu_stats['peak_util']:.0f}%[/]",
            "LLM decode is memory-BW bound — low util is normal"
        )
        table.add_row(
            "  VRAM usage",
            f"[{mem_color}]{gpu_stats['avg_mem_gb']:.1f} GB used / {gpu_stats['total_mem_gb']:.1f} GB ({mem_pct:.0f}%)[/]",
            "Weights + KV cache + activations"
        )
        if specs.get("mem_bw_gbs") and m.generation_tps > 0:
            # Rough effective BW: bytes moved per token ≈ model_params × dtype_bytes
            # We can't know exact model size here, so show as contextual note
            table.add_row(
                "  BW context",
                f"[dim]{m.generation_tps:.1f} tok/s on {specs['mem_bw_gbs']:.0f} GB/s bus[/]",
                "Throughput / bandwidth reveals memory efficiency"
            )

    elif gpu.backend == "apple":
        # Apple Silicon — static info only
        table.add_row(f"  [bold]{specs.get('chip', 'Apple Silicon')}[/]", "", "")
        if specs.get("gpu_cores"):
            table.add_row(
                "  GPU cores",
                specs["gpu_cores"],
                "Unified architecture — CPU + GPU share memory"
            )
        if specs.get("ram"):
            table.add_row(
                "  Unified memory",
                specs["ram"],
                "Shared by CPU + GPU — no separate VRAM pool"
            )
        table.add_row(
            "  Utilization",
            "[dim]Use Activity Monitor → GPU History[/]",
            "pynvml not supported on Apple Silicon"
        )
    else:
        table.add_row(
            "  GPU",
            "[dim]No GPU detected[/]",
            "Running on CPU — expect lower throughput"
        )

    return Panel(
        table,
        title=f"[bold white] Turn {turn} — Performance Metrics [/]",
        border_style="bright_blue",
        padding=(0, 1),
    )


def render_concept_legend() -> Panel:
    """One-time reference card shown at startup."""
    lines = [
        "[bold yellow]Tokens[/]            The atomic unit models process. ~¾ of a word on average.",
        "                   Input cost is O(n²) attention; output cost is O(n) per token.",
        "",
        "[bold yellow]TTFT[/]               Time To First Token. Dominated by prefill (prompt processing).",
        "                   Long prompts = slower TTFT even on fast GPUs.",
        "",
        "[bold yellow]Throughput[/]         Tokens/second during decode. Sequential — each token depends",
        "                   on the previous. Hard upper bound ≈ VRAM bandwidth / model size.",
        "",
        "[bold yellow]KV Cache[/]           Stores Key+Value attention tensors for all past tokens.",
        "                   Avoids recomputing attention on every new token. Grows with context.",
        "                   Memory = 2 × layers × heads × dim × seq_len × dtype_bytes",
        "",
        "[bold yellow]GPU Utilization[/]    LLM decode is memory-bandwidth bound, not compute bound.",
        "                   Even at 30% GPU util you may be at 100% memory bandwidth.",
    ]
    return Panel(
        "\n".join(lines),
        title="[bold white] Concept Reference [/]",
        border_style="dim",
        padding=(0, 2),
    )


def render_help() -> Panel:
    lines = [
        "[bold]/help[/]           Show this panel",
        "[bold]/clear[/]          Clear conversation history (frees KV cache context)",
        "[bold]/model[/]          Switch to a different Ollama model",
        "[bold]/ctx <n>[/]        Change context window size (e.g. /ctx 8192)",
        "[bold]/concepts[/]       Show the metric concept reference card",
        "[bold]/setup[/]           Pull the recommended demo models (llama3.2:1b, 3b, llama3.1:8b)",
        "[bold]/compare[/]        Run the same prompt across multiple models and compare metrics",
        "[bold]/tools[/]           Toggle Intersight MCP tools on/off",
        "[bold]/quit[/]           Exit",
        "",
        "[dim]Any other input is sent to the model.[/]",
    ]
    return Panel("\n".join(lines), title="[bold white] Commands [/]", border_style="dim", padding=(0, 2))


def render_comparison_summary(results: list[dict]) -> Panel:
    """
    Generates plain-English explanations of why each model won its best metric.
    Grounds the numbers in AI infrastructure concepts.
    """
    lines: list[str] = []

    def winner(key, lower_is_better=False):
        """Return (model_name, value) for the best result on a given key."""
        valid = [(r["model"], key(r["m"])) for r in results if key(r["m"]) > 0]
        if not valid:
            return None, None
        return min(valid, key=lambda x: x[1]) if lower_is_better else max(valid, key=lambda x: x[1])

    def info_tag(model_name: str) -> str:
        """Return a short '(7.6B Q4_K_M)' tag if info is available."""
        r = next((x for x in results if x["model"] == model_name), None)
        if not r:
            return ""
        info = r.get("info", {})
        parts = [p for p in [info.get("params", ""), info.get("quant", "")] if p]
        return f" ({', '.join(parts)})" if parts else ""

    # ── Time to First Token ───────────────────────────────────────────────────
    ttft_model, ttft_val = winner(lambda m: m.ttft if m.ttft > 0 else 0, lower_is_better=True)
    if ttft_model:
        tps_model, _ = winner(lambda m: m.generation_tps)
        same = ttft_model == tps_model
        lines.append(
            f"[bold green]{ttft_model}[/]{info_tag(ttft_model)} had the lowest time to first token "
            f"([bold]{ttft_val:.3f}s[/]). "
            + (
                "TTFT is dominated by prefill — processing all input tokens in parallel "
                "before generation begins. Smaller models load fewer weight matrices per "
                "attention layer, so each token costs less memory bandwidth during prefill."
                if same else
                "TTFT reflects prefill speed: how fast the model can digest the prompt. "
                "This model's architecture or quantisation level means fewer bytes moved "
                "per token during the initial forward pass, even if decode isn't the fastest."
            )
        )
        lines.append("")

    # ── Generation throughput ─────────────────────────────────────────────────
    tps_model, tps_val = winner(lambda m: m.generation_tps)
    if tps_model:
        lines.append(
            f"[bold green]{tps_model}[/]{info_tag(tps_model)} was the fastest at generation "
            f"([bold]{tps_val:.1f} tok/s[/]). "
            "Decode throughput is almost entirely determined by how many bytes the GPU "
            "must read from VRAM per token — larger models have more parameters, so each "
            "token requires more memory transfers. This is why a 1B model can be 3–5× "
            "faster than a 7B model even on the same GPU."
        )
        lines.append("")

    # ── Output length ─────────────────────────────────────────────────────────
    out_model, out_val = winner(lambda m: m.output_tokens)
    if out_model:
        lines.append(
            f"[bold green]{out_model}[/]{info_tag(out_model)} produced the most output tokens "
            f"([bold]{out_val:,}[/]). "
            "Larger models tend to generate longer, more detailed responses because they "
            "have seen more training data and have greater capacity to elaborate. More "
            "output tokens also means a longer decode phase — directly increasing total "
            "wall time and VRAM used by the growing KV cache."
        )
        lines.append("")

    # ── Prefill speed ─────────────────────────────────────────────────────────
    pre_model, pre_val = winner(lambda m: m.prompt_tps)
    if pre_model and pre_val > 0:
        lines.append(
            f"[bold green]{pre_model}[/]{info_tag(pre_model)} processed the prompt fastest "
            f"([bold]{pre_val:.0f} tok/s prefill[/]). "
            "Prefill is a batched matrix multiply — all prompt tokens are processed in "
            "parallel, so it scales well with GPU compute. Smaller models benefit here "
            "because each layer is cheaper, but highly optimised larger models can "
            "sometimes match them by saturating GPU tensor cores."
        )
        lines.append("")

    # ── GPU VRAM ──────────────────────────────────────────────────────────────
    gpu_results = [(r["model"], r["gpu"]) for r in results if r.get("gpu")]
    if gpu_results:
        lowest_vram = min(gpu_results, key=lambda x: x[1]["peak_mem_gb"])
        lines.append(
            f"[bold green]{lowest_vram[0]}[/]{info_tag(lowest_vram[0])} used the least VRAM "
            f"([bold]{lowest_vram[1]['peak_mem_gb']:.1f} GB peak[/]). "
            "VRAM footprint = model weights + KV cache + activations. Smaller or more "
            "aggressively quantised models (e.g. Q4 vs F16) shrink the weight footprint "
            "significantly. This matters for infrastructure planning: VRAM capacity sets "
            "a hard ceiling on which models you can serve and how many requests you can "
            "batch concurrently."
        )
        lines.append("")

    # ── General takeaway ──────────────────────────────────────────────────────
    lines.append("[dim]Key takeaway:[/] There is no universally best model — the right choice "
                 "depends on whether your workload is latency-sensitive (favour smaller/faster), "
                 "quality-sensitive (favour larger), or cost-sensitive (favour efficient quantisation). "
                 "These tradeoffs are the core of AI infrastructure design.")

    return Panel(
        "\n".join(lines),
        title="[bold white] Why These Results? [/]",
        border_style="bright_magenta",
        padding=(1, 2),
    )


def render_comparison_table(results: list[dict]) -> Panel:
    """
    Side-by-side metrics table across models for the same prompt.
    Great for showing how model size, architecture, and quantisation
    affect latency, throughput, and output quality.
    """
    table = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 1))
    table.add_column("Metric", style="bold cyan", width=24)
    for r in results:
        header = model_column_header(r["model"], r.get("info", {}))
        table.add_column(header, style="white", justify="right")

    def best(values, lower_is_better=False):
        """Return index of the best value."""
        valid = [(i, v) for i, v in enumerate(values) if v is not None]
        if not valid:
            return -1
        return min(valid, key=lambda x: x[1])[0] if lower_is_better else max(valid, key=lambda x: x[1])[0]

    def fmt_row(label, values, fmt, lower_is_better=False, unit=""):
        bi = best(values, lower_is_better)
        cells = []
        for i, v in enumerate(values):
            if v is None:
                cells.append("[dim]—[/]")
            else:
                text = f"{fmt(v)}{unit}"
                cells.append(f"[bold green]{text}[/]" if i == bi else text)
        table.add_row(label, *cells)

    # Tokens
    table.add_row("[bold yellow]TOKENS[/]", *[""] * len(results))
    fmt_row("  Input tokens",    [r["m"].prompt_tokens for r in results],  lambda v: f"{v:,}")
    fmt_row("  Output tokens",   [r["m"].output_tokens for r in results],  lambda v: f"{v:,}")

    # Latency
    table.add_row("[bold yellow]LATENCY[/]", *[""] * len(results))
    fmt_row("  Time to first token", [r["m"].ttft for r in results],       lambda v: f"{v:.3f}", lower_is_better=True, unit="s")
    fmt_row("  Total generation",    [r["m"].total_wall for r in results],  lambda v: f"{v:.2f}", lower_is_better=True, unit="s")

    # Throughput
    table.add_row("[bold yellow]THROUGHPUT[/]", *[""] * len(results))
    fmt_row("  Generation speed",    [r["m"].generation_tps for r in results], lambda v: f"{v:.1f}", unit=" tok/s")
    fmt_row("  Prefill speed",       [r["m"].prompt_tps for r in results],     lambda v: f"{v:.1f}", unit=" tok/s")

    # GPU static specs header (same GPU for all runs — show once as subtitle)
    # Dynamic per-run GPU stats
    gpu_stats = [r.get("gpu") for r in results]
    if any(s is not None for s in gpu_stats):
        first = next(s for s in gpu_stats if s)
        specs = results[0].get("gpu_specs", {})

        table.add_row("[bold yellow]HARDWARE[/]", *[""] * len(results))
        # Static specs — same across all runs, shown in first column only
        gpu_name = first.get("name", "GPU")
        arch     = specs.get("architecture", "")
        bw       = specs.get("mem_bw_gbs")
        vram     = first.get("total_mem_gb", 0)
        driver   = specs.get("driver", "")

        label_parts = [gpu_name]
        if arch:
            label_parts.append(arch)
        table.add_row(
            f"  {'  ·  '.join(label_parts)}",
            *[""] * len(results)
        )
        if bw:
            table.add_row(
                f"  Memory bandwidth",
                f"{bw:.0f} GB/s", *["[dim]same[/]"] * (len(results) - 1)
            )
        table.add_row(
            "  VRAM total",
            f"{vram:.1f} GB", *["[dim]same[/]"] * (len(results) - 1)
        )
        if driver:
            table.add_row(
                "  Driver",
                driver, *["[dim]same[/]"] * (len(results) - 1)
            )
        table.add_row("", *[""] * len(results))
        fmt_row(
            "  Avg GPU util",
            [s["avg_util"] if s else None for s in gpu_stats],
            lambda v: f"{v:.0f}", unit="%"
        )
        fmt_row(
            "  Peak VRAM used",
            [s["peak_mem_gb"] if s else None for s in gpu_stats],
            lambda v: f"{v:.1f}", unit=" GB"
        )

    return Panel(
        table,
        title="[bold white] Model Comparison — same prompt, different models [/]",
        subtitle="[dim green]green = best value[/]",
        border_style="bright_magenta",
        padding=(0, 1),
    )


def compare_models(prompt: str, num_ctx: int, gpu: GPUMonitor):
    """
    Run a single prompt against multiple models and render a comparison table.
    Each model gets a fresh single-turn context (no history) so results are comparable.
    """
    models = list_models()

    console.print()
    console.print(Rule("[bold]Select Models to Compare[/]"))
    console.print("[dim]Space-separated numbers, e.g:  1 3 4[/]\n")
    for i, name in enumerate(models, 1):
        console.print(f"  [cyan]{i}[/]. {name}")
    console.print()

    while True:
        raw = Prompt.ask("Models to compare")
        selected = []
        for tok in raw.split():
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(models):
                    selected.append(models[idx])
        if len(selected) >= 2:
            break
        console.print("[yellow]Pick at least 2 models.[/]")

    console.print()
    console.print(Panel(
        f"[bold]Prompt:[/] {escape(prompt)}",
        border_style="dim",
        padding=(0, 2),
    ))
    console.print()

    results = []
    for m_name in selected:
        console.print(Rule(f"[bold cyan]{m_name}[/]"))
        console.print(f"[bold cyan]Response:[/] ", end="")

        metrics = InferenceMetrics()
        chunks: list[str] = []

        gpu.start()
        metrics.t_request = time.perf_counter()

        try:
            stream = ollama.chat(
                model=m_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"num_ctx": num_ctx},
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    if not metrics.first_token_received:
                        metrics.t_first_token = time.perf_counter()
                        metrics.first_token_received = True
                    chunks.append(content)
                    console.print(content, end="", highlight=False)
                if chunk.get("done"):
                    metrics.t_done = time.perf_counter()
                    metrics.prompt_tokens    = chunk.get("prompt_eval_count", 0)
                    metrics.output_tokens    = chunk.get("eval_count", 0)
                    metrics.prompt_eval_ns   = chunk.get("prompt_eval_duration", 0)
                    metrics.eval_ns          = chunk.get("eval_duration", 0)
                    metrics.total_ns         = chunk.get("total_duration", 0)
                    metrics.load_ns          = chunk.get("load_duration", 0)
        except KeyboardInterrupt:
            metrics.t_done = time.perf_counter()
            console.print("\n[yellow][interrupted][/]")
        except Exception as e:
            metrics.t_done = time.perf_counter()
            console.print(f"\n[red]Error: {e}[/]")

        gpu.stop()
        console.print("\n")

        results.append({
            "model": m_name,
            "info": get_model_info(m_name),
            "m": metrics,
            "response": "".join(chunks),
            "gpu": gpu.stats(),
            "gpu_specs": gpu.specs,
        })

    console.print(render_comparison_table(results))
    console.print(render_comparison_summary(results))


# ── Model selection ───────────────────────────────────────────────────────────

def get_model_info(model_name: str) -> dict:
    """
    Fetch parameter count, quantisation, family, and disk size for a model.
    Returns a dict with keys: params, quant, family, size_gb.
    Falls back gracefully if the API doesn't return a field.
    """
    try:
        info = ollama.show(model_name)
        details = info.get("details", {})
        # Disk size comes from the model list entry
        size_bytes = 0
        for m in ollama.list().get("models", []):
            name = m.get("name") or m.get("model", "")
            if name == model_name:
                size_bytes = m.get("size", 0)
                break
        return {
            "params": details.get("parameter_size", ""),       # e.g. "7.6B"
            "quant":  details.get("quantization_level", ""),   # e.g. "Q4_K_M"
            "family": details.get("family", ""),               # e.g. "qwen2"
            "size_gb": size_bytes / 1024**3 if size_bytes else 0.0,
        }
    except Exception:
        return {"params": "", "quant": "", "family": "", "size_gb": 0.0}


def model_column_header(model_name: str, info: dict) -> str:
    """Build a compact multi-line column header with model characteristics."""
    parts = []
    if info["params"]:
        parts.append(info["params"])
    if info["quant"]:
        parts.append(info["quant"])
    if info["size_gb"] > 0:
        parts.append(f"{info['size_gb']:.1f} GB")
    subtitle = " · ".join(parts) if parts else ""
    short_name = model_name.split("/")[-1]  # strip registry prefix if present
    return f"[bold]{short_name}[/]\n[dim]{subtitle}[/]" if subtitle else f"[bold]{short_name}[/]"


def list_models() -> list[str]:
    try:
        models = ollama.list()
        names = []
        for m in models.get("models", []):
            name = m.get("name") or m.get("model", "")
            if name:
                names.append(name)
        return names
    except Exception as e:
        console.print(f"[red]Could not reach Ollama: {e}[/]")
        console.print("[dim]Make sure Ollama is running: [bold]ollama serve[/][/]")
        sys.exit(1)


SUGGESTED_MODELS = [
    ("llama3.2:1b",  "~700 MB  — fastest, good for demos on limited hardware"),
    ("llama3.2",     "~2 GB    — balanced speed and quality"),
    ("llama3.1:8b",  "~4.7 GB  — stronger reasoning"),
    ("qwen2.5:7b",   "~4.7 GB  — strong alternative to llama"),
    ("mistral",      "~4.1 GB  — fast, good instruction following"),
]

# ── Demo model sets — chosen based on available VRAM ─────────────────────────
#
# STANDARD (< 24 GB VRAM / Apple Silicon / CPU)
#   Same Llama 3 family at three sizes — isolates parameter count as the variable.
#
# HIGH-END (>= 24 GB VRAM, e.g. L40S / A100 / H100)
#   8B vs 70B (same family, dramatic size gap) + Qwen 72B (different architecture)
#   to show that model family also matters at the same parameter scale.

DEMO_MODELS_STANDARD: list[tuple[str, str, str]] = [
    ("llama3.2:1b", "~700 MB", "1B params — baseline speed, minimal VRAM"),
    ("llama3.2:3b", "~2 GB",   "3B params — quality step up, still fast"),
    ("llama3.1:8b", "~4.7 GB", "8B params — strongest reasoning, highest VRAM"),
]

DEMO_MODELS_HIGH_END: list[tuple[str, str, str]] = [
    ("llama3.1:8b",   "~4.7 GB", "8B params — fast baseline, barely touches L40S bandwidth"),
    ("llama3.1:70b",  "~40 GB",  "70B params — same family, saturates VRAM bandwidth"),
    ("qwen2.5:72b",   "~44 GB",  "72B params — different architecture, compare tok/s at same scale"),
]

# VRAM threshold (GB) for switching to the high-end set
_HIGH_END_VRAM_GB = 24


def _select_demo_models(gpu: "GPUMonitor") -> tuple[list[tuple[str, str, str]], str]:
    """Return the appropriate demo model set and a label based on detected VRAM."""
    vram = gpu.total_mem_gb if gpu.available else 0
    if vram >= _HIGH_END_VRAM_GB:
        return DEMO_MODELS_HIGH_END, f"High-End GPU ({vram:.0f} GB VRAM detected)"
    return DEMO_MODELS_STANDARD, "Standard (< 24 GB VRAM / Apple Silicon / CPU)"


# Keep a single alias used by the rest of the code — resolved at runtime via _select_demo_models
DEMO_MODELS = DEMO_MODELS_STANDARD  # default; overridden after GPU detection


def pull_demo_models(
    models: list[tuple[str, str, str]] | None = None,
    label: str = "",
):
    """Pull the recommended demo models for the detected hardware, skipping any already installed."""
    if models is None:
        models = DEMO_MODELS
    installed = set(list_models())

    to_pull = [(n, s, d) for n, s, d in models if n not in installed]

    if not to_pull:
        console.print("[green]All demo models are already installed.[/]")
        return

    tier_note = (
        "These models span 8B → 70B+ parameters across two architectures —\n"
        "designed to saturate high-VRAM GPUs and show clear throughput differences."
        if models is DEMO_MODELS_HIGH_END else
        "These three models are the same architecture (Llama 3) at different sizes.\n"
        "Keeping architecture constant lets you isolate the effect of parameter count\n"
        "on speed, VRAM, and quality — the core infrastructure tradeoff."
    )

    console.print()
    console.print(Panel(
        f"[bold white]Demo Model Set[/]  [dim]{label}[/]\n\n" + tier_note,
        border_style="bright_blue",
        padding=(0, 2),
    ))
    console.print()

    for name, size, desc in models:
        if name in installed:
            console.print(f"  [green]✓[/] [bold]{name}[/]  [dim]{size} — already installed[/]")
        else:
            console.print(f"  [yellow]↓[/] [bold]{name}[/]  [dim]{size} — {desc}[/]")

    console.print()
    gb_to_pull = sum(
        float(s.replace("~", "").replace(" GB", ""))
        for _, s, _ in to_pull if "GB" in s
    )
    confirm = Prompt.ask(
        f"Pull {len(to_pull)} model(s)? (~{gb_to_pull:.1f} GB download)",
        choices=["y", "n"],
        default="y",
    )
    if confirm != "y":
        return

    console.print()
    for name, size, desc in to_pull:
        if name in installed:
            continue
        console.print(Rule(f"[bold cyan]Pulling {name}[/] [dim]{size}[/]"))
        result = subprocess.run(["ollama", "pull", name])
        if result.returncode == 0:
            console.print(f"[green]Done:[/] {name}\n")
        else:
            console.print(f"[red]Failed to pull {name}[/] — skipping.\n")


def pull_model_interactively():
    """Prompt the user to pick a model to pull when none are installed."""
    console.print()
    console.print(Panel(
        "[yellow]No models are installed.[/]\n\n"
        "Ollama needs at least one model to chat with.\n"
        "Select one below to download it now.",
        border_style="yellow",
        padding=(0, 2),
    ))
    console.print()
    console.print(Rule("[bold]Recommended Models[/]"))
    for i, (name, note) in enumerate(SUGGESTED_MODELS, 1):
        console.print(f"  [cyan]{i}[/]. [bold]{name}[/]  [dim]{note}[/]")
    console.print(f"  [cyan]{len(SUGGESTED_MODELS) + 1}[/]. [dim]Enter a custom model name[/]")
    console.print()

    while True:
        choice = Prompt.ask("Select a model to pull")
        if choice.isdigit():
            idx = int(choice) - 1
            if idx == len(SUGGESTED_MODELS):
                model_name = Prompt.ask("Model name (e.g. phi3, gemma2:2b)")
            elif 0 <= idx < len(SUGGESTED_MODELS):
                model_name = SUGGESTED_MODELS[idx][0]
            else:
                console.print("[yellow]Invalid choice, try again.[/]")
                continue
        else:
            model_name = choice.strip()

        if not model_name:
            continue

        console.print(f"\nPulling [bold]{model_name}[/] — this may take a few minutes...")
        console.print("[dim]Progress is shown by Ollama below.[/]\n")
        result = subprocess.run(["ollama", "pull", model_name])
        if result.returncode == 0:
            console.print(f"\n[green]Done![/] {model_name} is ready.")
            return
        else:
            console.print(f"[red]Pull failed.[/] Check the model name and try again.")


def pick_model(current: str | None = None) -> tuple[str, int]:
    models = list_models()
    if not models:
        pull_model_interactively()
        models = list_models()  # refresh after pull
    if not models:
        console.print("[red]Still no models found — exiting.[/]")
        sys.exit(1)

    console.print()
    console.print(Rule("[bold]Available Models[/]"))
    for i, name in enumerate(models, 1):
        marker = " [green]←[/]" if name == current else ""
        console.print(f"  [cyan]{i}[/]. {name}{marker}")
    console.print()

    while True:
        choice = Prompt.ask("Select model number (or type name)", default=models[0])
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                break
        elif choice in models:
            selected = choice
            break
        else:
            console.print("[yellow]Not found, try again.[/]")

    ctx_default = 4096
    ctx_str = Prompt.ask(
        "Context window size [dim](num_ctx)[/]",
        default=str(ctx_default)
    )
    try:
        ctx = int(ctx_str)
    except ValueError:
        ctx = ctx_default

    return selected, ctx


# ── Main chat loop ────────────────────────────────────────────────────────────

def chat_turn(
    model: str,
    messages: list[dict],
    num_ctx: int,
    gpu: GPUMonitor,
    turn: int,
) -> tuple[str, InferenceMetrics]:
    """Stream one turn, collect metrics, return (response_text, metrics)."""
    m = InferenceMetrics()
    response_chunks: list[str] = []

    gpu.start()
    m.t_request = time.perf_counter()

    console.print()
    console.print(Rule(f"[dim]turn {turn}[/]"))
    console.print(f"[bold cyan]Assistant:[/] ", end="")

    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True,
            options={"num_ctx": num_ctx},
        )

        for chunk in stream:
            msg = chunk.get("message", {})
            content = msg.get("content", "")

            if content:
                if not m.first_token_received:
                    m.t_first_token = time.perf_counter()
                    m.first_token_received = True
                response_chunks.append(content)
                console.print(content, end="", highlight=False)

            # Final chunk carries the statistics
            if chunk.get("done"):
                m.t_done = time.perf_counter()
                m.prompt_tokens = chunk.get("prompt_eval_count", 0)
                m.output_tokens = chunk.get("eval_count", 0)
                m.prompt_eval_ns = chunk.get("prompt_eval_duration", 0)
                m.eval_ns = chunk.get("eval_duration", 0)
                m.total_ns = chunk.get("total_duration", 0)
                m.load_ns = chunk.get("load_duration", 0)

    except KeyboardInterrupt:
        m.t_done = time.perf_counter()
        console.print("\n[yellow][interrupted][/]")
    except Exception as e:
        m.t_done = time.perf_counter()
        console.print(f"\n[red]Error: {e}[/]")

    gpu.stop()
    console.print()

    return "".join(response_chunks), m


def chat_turn_with_tools(
    model: str,
    messages: list[dict],
    num_ctx: int,
    gpu: GPUMonitor,
    turn: int,
    mcp: MCPClient,
) -> tuple[str, InferenceMetrics, list[dict]]:
    """
    Tool-aware chat turn. Flow:
      1. Send to Ollama with tools — non-streaming so it can decide to call tools
      2. If tool calls returned: execute each via MCP, show inline, loop back
      3. Stream the final response once tools are satisfied
    Returns (response_text, metrics, tool_call_log).

    This shows the 'model knows vs tool needed' distinction:
      - "What is Intersight?" → answers from training, no tool call
      - "How many servers do we have?" → calls get_environment_summary()
    """
    m = InferenceMetrics()
    tool_call_log: list[dict] = []

    gpu.start()
    m.t_request = time.perf_counter()

    console.print()
    console.print(Rule(f"[dim]turn {turn}[/]"))

    try:
        # ── Phase 1: let the model decide if it needs tools ───────────────────
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=mcp.tools,
            options={"num_ctx": num_ctx},
            stream=False,
        )

        # ── Phase 2: execute tool calls if any ───────────────────────────────
        tool_messages = []
        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                fn = tc.function
                args = fn.arguments if isinstance(fn.arguments, dict) else {}

                console.print(
                    f"[bold magenta]⚙ Tool call:[/] [cyan]{fn.name}[/]"
                    + (f"  [dim]{args}[/]" if args else "")
                )

                t_tool_start = time.perf_counter()
                result = mcp.call(fn.name, args)
                t_tool_end = time.perf_counter()

                tool_call_log.append({
                    "name": fn.name,
                    "args": args,
                    "duration": t_tool_end - t_tool_start,
                    "result_preview": result[:120] + "…" if len(result) > 120 else result,
                })

                console.print(f"[dim]  → {result[:200]}{'…' if len(result) > 200 else ''}[/]")
                console.print()

                tool_messages.append({
                    "role": "tool",
                    "content": result,
                })

            # Add assistant's tool-call message + results to context
            messages = messages + [
                {"role": "assistant", "content": response.message.content or "",
                 "tool_calls": response.message.tool_calls}
            ] + tool_messages

        # ── Phase 3: stream the final response ────────────────────────────────
        console.print(f"[bold cyan]Assistant:[/] ", end="")
        response_chunks: list[str] = []

        stream = ollama.chat(
            model=model,
            messages=messages,
            options={"num_ctx": num_ctx},
            stream=True,
        )

        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                if not m.first_token_received:
                    m.t_first_token = time.perf_counter()
                    m.first_token_received = True
                response_chunks.append(content)
                console.print(content, end="", highlight=False)
            if chunk.get("done"):
                m.t_done = time.perf_counter()
                m.prompt_tokens = chunk.get("prompt_eval_count", 0)
                m.output_tokens = chunk.get("eval_count", 0)
                m.prompt_eval_ns = chunk.get("prompt_eval_duration", 0)
                m.eval_ns = chunk.get("eval_duration", 0)
                m.total_ns = chunk.get("total_duration", 0)
                m.load_ns = chunk.get("load_duration", 0)

    except KeyboardInterrupt:
        m.t_done = time.perf_counter()
        console.print("\n[yellow][interrupted][/]")
    except Exception as e:
        m.t_done = time.perf_counter()
        console.print(f"\n[red]Error: {e}[/]")

    gpu.stop()
    console.print()

    return "".join(response_chunks), m, tool_call_log


def render_tool_call_panel(tool_calls: list[dict]) -> Panel:
    """Shows which tools were called, their arguments, timing, and result preview."""
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("", style="bold magenta", width=20)
    table.add_column("", style="white")

    for i, tc in enumerate(tool_calls, 1):
        table.add_row(f"  Tool {i}", f"[bold cyan]{tc['name']}[/]")
        if tc["args"]:
            table.add_row("  Args", f"[dim]{tc['args']}[/]")
        table.add_row("  Time", f"{tc['duration']:.2f}s  [dim](Intersight API round-trip)[/]")
        table.add_row("  Result", f"[dim]{tc['result_preview']}[/]")
        if i < len(tool_calls):
            table.add_row("", "")

    return Panel(
        table,
        title="[bold white] MCP Tool Calls — live Intersight data [/]",
        border_style="magenta",
        padding=(0, 1),
    )


def main():
    ensure_ollama()
    console.clear()
    console.print(Panel(
        "[bold white]AI Fundamentals Demo[/]\n"
        "[dim]Interactive CLI chat with real-time performance metrics.\n"
        "Watch how tokens, latency, KV cache, and GPU utilization change\n"
        "as you vary prompt length, conversation depth, and model size.[/]",
        border_style="bright_blue",
        padding=(1, 4),
    ))
    console.print()
    console.print(render_concept_legend())
    console.print()

    # Initialise GPU monitor first — needed to detect VRAM for model tier selection
    gpu = GPUMonitor()

    # Select the right demo model set based on detected VRAM
    demo_models, demo_label = _select_demo_models(gpu)

    # Offer to pull the recommended demo model set if any are missing
    installed = set(list_models())
    missing_demo = [name for name, _, _ in demo_models if name not in installed]
    if missing_demo:
        console.print(Panel(
            f"[yellow]{len(missing_demo)} of the 3 recommended demo models are not installed.[/]\n"
            f"Tier: [bold]{demo_label}[/]\n"
            "Pulling them enables the [bold]/compare[/] command to show clear size vs speed tradeoffs.\n\n"
            f"Missing: [bold]{', '.join(missing_demo)}[/]",
            border_style="yellow",
            padding=(0, 2),
        ))
        setup = Prompt.ask("Pull recommended demo models now?", choices=["y", "n"], default="y")
        if setup == "y":
            pull_demo_models(demo_models, demo_label)
        console.print()

    model, num_ctx = pick_model()

    if gpu.available:
        console.print(f"[green]GPU monitoring:[/] {gpu.gpu_name} ({gpu.total_mem_gb:.1f} GB)")
    else:
        console.print("[dim]GPU monitoring: not available (install pynvml for NVIDIA stats)[/]")

    console.print(f"[green]Model:[/] {model}  |  [green]Context window:[/] {num_ctx:,} tokens")
    console.print(f"[dim]Type /help for commands.[/]")
    console.print()

    # ── MCP / Intersight tools ────────────────────────────────────────────────
    mcp_client = MCPClient()
    tools_enabled = False

    if os.path.exists(".env") or os.environ.get("INTERSIGHT_API_KEY_ID"):
        console.print("[dim]Intersight credentials detected. Use [bold]/tools[/] to enable MCP tool calling.[/]")
    else:
        console.print(
            "[dim]No Intersight credentials found. "
            "Copy [bold].env.example[/] to [bold].env[/] to enable MCP tools.[/]"
        )
    console.print()

    messages: list[dict] = []
    turn = 0
    total_ctx_tokens = 0
    # Tracks the true context length using actual Ollama token counts, not estimates.
    # After each turn: true_ctx = cached_prefix + newly_evaluated + output_tokens
    true_context_tokens = 0

    while True:
        try:
            prompt_label = "[bold green]Prompt[/] [magenta][tools][/]" if tools_enabled else "[bold green]Prompt[/]"
            user_input = Prompt.ask(prompt_label)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if not user_input.strip():
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        cmd = user_input.strip().lower()

        if cmd in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye.[/]")
            break

        if cmd == "/help":
            console.print(render_help())
            continue

        if cmd == "/concepts":
            console.print(render_concept_legend())
            continue

        if cmd == "/clear":
            messages.clear()
            total_ctx_tokens = 0
            true_context_tokens = 0
            turn = 0
            console.print(
                "[yellow]History cleared.[/] [dim]KV cache context reset — "
                "next prompt will be processed from scratch.[/]"
            )
            continue

        if cmd == "/setup":
            _models, _label = _select_demo_models(gpu)
            pull_demo_models(_models, _label)
            continue

        if cmd == "/tools":
            if tools_enabled:
                tools_enabled = False
                console.print("[yellow]MCP tools disabled.[/] Responses use model training data only.")
            else:
                console.print("[dim]Connecting to Intersight MCP server...[/]")
                if mcp_client.setup():
                    tools_enabled = True
                    tool_names = [t["function"]["name"] for t in mcp_client.tools]
                    console.print(Panel(
                        "[green]MCP tools enabled.[/] The model can now call live Intersight data.\n\n"
                        f"Available tools: [cyan]{', '.join(tool_names)}[/]\n\n"
                        "[dim]Try asking:\n"
                        "  • 'How many servers do we have?' — needs a tool\n"
                        "  • 'Are there any critical alarms?' — needs a tool\n"
                        "  • 'What is Intersight?' — model answers from training data, no tool needed[/]",
                        border_style="magenta",
                        padding=(0, 2),
                    ))
                else:
                    console.print(
                        f"[red]Failed to connect to Intersight MCP server.[/]\n"
                        f"[dim]{mcp_client.error}[/]\n\n"
                        "Check that [bold].env[/] exists with valid Intersight credentials."
                    )
            continue

        if cmd == "/model":
            model, num_ctx = pick_model(current=model)
            messages.clear()
            total_ctx_tokens = 0
            true_context_tokens = 0
            turn = 0
            console.print(f"[green]Switched to:[/] {model}  |  context: {num_ctx:,} tokens")
            continue

        if cmd == "/compare":
            # Use the last user message as the prompt, or ask for one
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                None,
            )
            if last_user:
                use_last = Prompt.ask(
                    f"Use last prompt? [dim]\"{escape(last_user[:60])}{'...' if len(last_user) > 60 else ''}\"[/]",
                    choices=["y", "n"],
                    default="y",
                )
                compare_prompt = last_user if use_last == "y" else Prompt.ask("Enter prompt to compare")
            else:
                compare_prompt = Prompt.ask("Enter prompt to compare")
            compare_models(compare_prompt, num_ctx, gpu)
            continue

        if cmd.startswith("/ctx "):
            try:
                num_ctx = int(cmd.split()[1])
                console.print(f"[green]Context window set to:[/] {num_ctx:,} tokens")
            except (IndexError, ValueError):
                console.print("[red]Usage: /ctx 8192[/]")
            continue

        # ── Regular chat turn ─────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})
        turn += 1

        # Save the true context length before this turn so we can measure cache reuse.
        # true_context_tokens is the sum of actual Ollama-reported tokens from all
        # prior turns — not a word-count estimate.
        prev_context_tokens = true_context_tokens

        if tools_enabled and mcp_client.available:
            response, metrics, tool_calls = chat_turn_with_tools(
                model, messages, num_ctx, gpu, turn, mcp_client
            )
            if tool_calls:
                console.print(render_tool_call_panel(tool_calls))
        else:
            response, metrics = chat_turn(model, messages, num_ctx, gpu, turn)
            tool_calls = []

        if response:
            messages.append({"role": "assistant", "content": response})

        # ── KV cache reuse (accurate) ──────────────────────────────────────────
        # Ollama's prompt_eval_count = tokens it actually re-evaluated this turn.
        # If prefix caching hit, prompt_eval_count ≈ just the new user message.
        # If no cache hit, prompt_eval_count ≈ entire conversation history.
        #
        # cached = tokens Ollama skipped  = prev context − newly evaluated
        # (clamped to 0 when prompt_eval > prev, i.e. first turn or after /clear)
        cached_tokens = max(0, prev_context_tokens - metrics.prompt_tokens)

        # True context length = the prefix Ollama reused + tokens it evaluated
        # this turn + the response it generated.
        raw_context = cached_tokens + metrics.prompt_tokens + metrics.output_tokens

        # If raw_context exceeds num_ctx, Ollama silently truncated the oldest
        # tokens to fit the window. Cap to num_ctx so the usage bar stays accurate
        # and flag truncation so we can warn the user.
        truncated = raw_context > num_ctx and turn > 1
        true_context_tokens = min(raw_context, num_ctx)
        total_ctx_tokens = true_context_tokens

        console.print(render_metrics_panel(
            metrics,
            num_ctx=num_ctx,
            total_ctx_tokens=total_ctx_tokens,
            gpu=gpu,
            turn=turn,
            cached_tokens=cached_tokens,
        ))

        ctx_pct = total_ctx_tokens / num_ctx * 100 if num_ctx > 0 else 0

        if truncated:
            console.print(Panel(
                f"[red]Context window overflow.[/] Ollama silently dropped the oldest tokens to fit within "
                f"[bold]{num_ctx:,}[/] tokens.\n\n"
                "The model is still responding but has [bold]lost the beginning of the conversation[/] — "
                "it may contradict or forget earlier context without warning.\n\n"
                f"[dim]Run [bold]/ctx {num_ctx * 2:,}[/] to double the window, "
                "or [bold]/clear[/] to start fresh.[/]",
                border_style="red",
                padding=(0, 2),
            ))
        elif ctx_pct > 80:
            console.print(
                f"[yellow]Warning:[/] Context {ctx_pct:.0f}% full. "
                "Use [bold]/clear[/] to reset or [bold]/ctx[/] to increase the window."
            )


if __name__ == "__main__":
    main()
