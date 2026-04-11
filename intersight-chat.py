#!/usr/bin/env python3
"""
Intersight AI Demo
==================
A focused demo that connects a local LLM (via Ollama) to live Cisco Intersight
infrastructure data using MCP tool calling.

Shows the key AI concept: the difference between what a model *knows*
(training data) vs. what it can *do* (live tool access via MCP).

Usage:
    ./intersight-chat.sh          # recommended — handles venv + deps
    python intersight-chat.py     # if deps are already installed
"""

import os
import sys
import time
import subprocess
import platform
import asyncio

# ── Dependency check ──────────────────────────────────────────────────────────

def _check_deps():
    missing = []
    for pkg, imp in [
        ("ollama", "ollama"),
        ("rich", "rich"),
        ("mcp", "mcp"),
        ("intersight", "intersight"),
        ("python-dotenv", "dotenv"),
        ("certifi", "certifi"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Run: pip install {' '.join(missing)}\n")
        sys.exit(1)

_check_deps()

import ollama
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich import box
from dotenv import load_dotenv

load_dotenv()

console = Console()

# ── MCP client ────────────────────────────────────────────────────────────────

MCP_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intersight_mcp.py")

class MCPClient:
    def __init__(self):
        self.tools: list[dict] = []
        self.available = False
        self.error = ""

    def setup(self) -> bool:
        try:
            self.tools = asyncio.run(self._fetch_tools())
            self.available = bool(self.tools)
            return self.available
        except Exception as e:
            self.error = str(e)
            return False

    def call(self, name: str, arguments: dict) -> str:
        try:
            return asyncio.run(self._call_tool(name, arguments))
        except Exception as e:
            return f"Tool error: {e}"

    async def _fetch_tools(self) -> list[dict]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        env = {**os.environ, "PYTHONPATH": os.path.dirname(MCP_SERVER)}
        params = StdioServerParameters(command=sys.executable, args=[MCP_SERVER], env=env)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": t.inputSchema or {"type": "object", "properties": {}},
                        },
                    }
                    for t in result.tools
                ]

    async def _call_tool(self, name: str, arguments: dict) -> str:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        env = {**os.environ, "PYTHONPATH": os.path.dirname(MCP_SERVER)}
        params = StdioServerParameters(command=sys.executable, args=[MCP_SERVER], env=env)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return "\n".join(
                    c.text for c in result.content if hasattr(c, "text")
                ) or "(no output)"


# ── Ollama helpers ─────────────────────────────────────────────────────────────

def ensure_ollama() -> bool:
    """Check Ollama is running, start it if not."""
    try:
        ollama.list()
        return True
    except Exception:
        pass
    console.print("[dim]Starting Ollama...[/]")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        ollama.list()
        return True
    except Exception as e:
        console.print(f"[red]Could not start Ollama:[/] {e}")
        return False


def pick_model() -> str:
    """Let the user choose from installed models, with a recommended default."""
    try:
        models = [m.model for m in ollama.list().models]
    except Exception:
        models = []

    if not models:
        console.print("[yellow]No models installed.[/] Pull one with: [bold]ollama pull llama3.1:8b[/]")
        sys.exit(1)

    # Prefer tool-capable models in this order
    preferred = ["llama3.1:8b", "llama3.2:3b", "llama3.2:1b", "qwen2.5:7b", "mistral:7b"]
    default = next((m for m in preferred if m in models), models[0])

    console.print("\n[bold]Available models:[/]")
    for i, m in enumerate(models, 1):
        tag = " [green](recommended)[/]" if m == default else ""
        console.print(f"  [cyan]{i}.[/] {m}{tag}")

    console.print()
    choice = Prompt.ask("Select model", default=default)

    # Accept number or name
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
    if choice in models:
        return choice
    console.print(f"[yellow]'{choice}' not found, using {default}[/]")
    return default


# ── Chat ──────────────────────────────────────────────────────────────────────

def build_system_prompt(mcp: MCPClient) -> dict:
    tool_names = ", ".join(
        t["function"]["name"] for t in mcp.tools[:12]
    ) + (" …and more" if len(mcp.tools) > 12 else "")

    return {
        "role": "system",
        "content": (
            "You are an infrastructure assistant with live access to Cisco Intersight "
            "via tool calls.\n\n"
            "RULES:\n"
            "1. ALWAYS use the provided tools to answer questions about servers, alarms, "
            "firmware, policies, fabric, hardware inventory, or any real-time infrastructure "
            "state. Never guess or use training data for live facts.\n"
            "2. Only answer from training data for conceptual questions like "
            "'What is Intersight?' or 'How does UCS work?'\n"
            "3. When listing results, summarize clearly — total counts, key fields, "
            "and any notable findings.\n\n"
            f"Available tools include: {tool_names}.\n\n"
            "Examples:\n"
            "- 'How many servers do we have?' → call list_compute_servers\n"
            "- 'Any critical alarms?' → call list_alarms with severity=Critical\n"
            "- 'What firmware is running?' → call get_firmware_summary\n"
            "- 'List the UCS domains' → call list_fabric_interconnects"
        ),
    }


def chat_turn(model: str, messages: list[dict], mcp: MCPClient) -> str:
    """Single tool-aware chat turn. Returns the assistant's final response text."""
    system_msg = build_system_prompt(mcp)
    full_messages = [system_msg] + messages

    # Phase 1 — let model decide if it needs tools (non-streaming)
    response = ollama.chat(
        model=model,
        messages=full_messages,
        tools=mcp.tools,
        stream=False,
    )

    # Phase 2 — execute any tool calls
    if response.message.tool_calls:
        for tc in response.message.tool_calls:
            fn = tc.function
            args = fn.arguments if isinstance(fn.arguments, dict) else {}

            console.print(
                f"\n[bold magenta]⚙ Tool call:[/] [cyan]{fn.name}[/]"
                + (f"  [dim]{args}[/]" if args else "")
            )

            t0 = time.perf_counter()
            result = mcp.call(fn.name, args)
            elapsed = time.perf_counter() - t0

            console.print(f"[dim]  → {result[:300]}{'…' if len(result) > 300 else ''}[/]")
            console.print(f"[dim]  ({elapsed:.2f}s)[/]\n")

        # Add assistant tool-call turn + tool results to context
        full_messages = full_messages + [
            {
                "role": "assistant",
                "content": response.message.content or "",
                "tool_calls": response.message.tool_calls,
            }
        ] + [
            {"role": "tool", "content": mcp.call(tc.function.name,
                tc.function.arguments if isinstance(tc.function.arguments, dict) else {})}
            for tc in response.message.tool_calls
        ]

    # Phase 3 — stream the final answer
    console.print("[bold cyan]Assistant:[/] ", end="")
    chunks: list[str] = []
    stream = ollama.chat(model=model, messages=full_messages, stream=True)
    for chunk in stream:
        token = chunk.message.content or ""
        console.print(token, end="", markup=False)
        chunks.append(token)
    console.print()
    return "".join(chunks)


# ── Help & UI ─────────────────────────────────────────────────────────────────

HELP = """
[bold]Commands[/]
  [cyan]/clear[/]    Clear conversation history
  [cyan]/model[/]    Switch to a different model
  [cyan]/tools[/]    Show available Intersight tools
  [cyan]/help[/]     Show this message
  [cyan]/quit[/]     Exit
"""

def show_welcome(model: str, mcp: MCPClient):
    tool_mode = os.environ.get("INTERSIGHT_TOOL_MODE", "core").upper()
    console.print(Panel(
        f"[bold green]Intersight AI Demo[/]\n\n"
        f"Model: [cyan]{model}[/]\n"
        f"Tools: [green]{len(mcp.tools)} Intersight tools loaded[/] "
        f"[dim]({tool_mode} mode)[/]\n\n"
        "[dim]Ask about your infrastructure — servers, alarms, firmware, fabric.\n"
        "The model will call live Intersight tools to answer.\n\n"
        "Try: [bold]list servers[/]  |  [bold]any critical alarms?[/]  |  "
        "[bold]what firmware is running?[/][/]",
        border_style="green",
        padding=(1, 2),
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold]Intersight AI Demo[/]")
    console.print("[dim]Connecting to Ollama...[/]")

    if not ensure_ollama():
        sys.exit(1)

    model = pick_model()

    console.print("\n[dim]Loading Intersight MCP tools...[/]")
    mcp = MCPClient()
    if not mcp.setup():
        console.print(Panel(
            f"[red]Could not connect to Intersight MCP server.[/]\n\n"
            f"Error: {mcp.error}\n\n"
            "Make sure your [bold].env[/] file has valid credentials:\n"
            "  [cyan]INTERSIGHT_CLIENT_ID[/]=your_client_id\n"
            "  [cyan]INTERSIGHT_CLIENT_SECRET[/]=your_client_secret\n\n"
            "See [bold].env.example[/] for all options.",
            border_style="red",
            padding=(1, 2),
        ))
        sys.exit(1)

    show_welcome(model, mcp)

    messages: list[dict] = []

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()

        if cmd in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye.[/]")
            break

        if cmd == "/help":
            console.print(HELP)
            continue

        if cmd == "/clear":
            messages.clear()
            console.print("[green]Conversation cleared.[/]")
            continue

        if cmd == "/model":
            model = pick_model()
            messages.clear()
            console.print(f"[green]Switched to {model}. Conversation cleared.[/]")
            continue

        if cmd == "/tools":
            t = Table(title=f"Intersight Tools ({len(mcp.tools)})", box=box.SIMPLE)
            t.add_column("Tool", style="cyan")
            t.add_column("Description")
            for tool in mcp.tools:
                fn = tool["function"]
                t.add_row(fn["name"], (fn.get("description") or "")[:80])
            console.print(t)
            continue

        # Regular chat turn
        messages.append({"role": "user", "content": user_input})
        console.print(Rule("[dim]thinking[/]"))

        try:
            response = chat_turn(model, messages, mcp)
            if response:
                messages.append({"role": "assistant", "content": response})
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            messages.pop()  # remove the user message that caused the error


if __name__ == "__main__":
    main()
