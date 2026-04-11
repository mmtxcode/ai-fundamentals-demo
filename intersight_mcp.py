#!/usr/bin/env python3
"""
Intersight MCP Server
=====================
Exposes Cisco Intersight infrastructure data as tools for local LLMs via MCP.

This lets a local model answer real-time infrastructure questions by calling
these tools, rather than relying on (potentially outdated) training data.

Setup:
  cp .env.example .env
  # Edit .env with your Intersight API key credentials
  python intersight_mcp.py        # run standalone for testing
  # chat.py starts this automatically when /tools is enabled
"""

import os
import sys


def _check_deps():
    missing = []
    for pkg in ("mcp", "intersight", "dotenv"):
        try:
            __import__(pkg if pkg != "dotenv" else "dotenv")
        except ImportError:
            missing.append("mcp" if pkg == "mcp" else
                           "intersight" if pkg == "intersight" else
                           "python-dotenv")
    if missing:
        print(f"Missing packages: {', '.join(missing)}", file=sys.stderr)
        print(f"Run: pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)


_check_deps()

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Intersight Infrastructure",
    instructions=(
        "Use these tools to answer questions about the live Intersight environment. "
        "Always call a tool when asked about specific servers, alarms, firmware, or "
        "infrastructure state — do not guess or use training data for real-time facts."
    ),
)


# ── Authentication ────────────────────────────────────────────────────────────

def _get_client():
    """
    Build an authenticated Intersight ApiClient.

    Required env vars (set in .env):
      INTERSIGHT_API_KEY_ID          — your API key ID
      INTERSIGHT_API_SECRET_KEY_FILE — path to your private key PEM file
        OR
      INTERSIGHT_API_SECRET_KEY      — PEM key content directly as a string

    Optional:
      INTERSIGHT_BASE_URL            — defaults to https://intersight.com
    """
    import intersight
    from intersight.api_client import ApiClient

    key_id = os.environ.get("INTERSIGHT_API_KEY_ID", "").strip()
    key_file = os.environ.get("INTERSIGHT_API_SECRET_KEY_FILE", "").strip()
    key_string = os.environ.get("INTERSIGHT_API_SECRET_KEY", "").strip()
    base_url = os.environ.get("INTERSIGHT_BASE_URL", "https://intersight.com").strip()

    if not key_id:
        raise RuntimeError(
            "INTERSIGHT_API_KEY_ID is not set. "
            "Copy .env.example to .env and add your credentials."
        )
    if not key_file and not key_string:
        raise RuntimeError(
            "Set INTERSIGHT_API_SECRET_KEY_FILE (path to PEM) "
            "or INTERSIGHT_API_SECRET_KEY (PEM content) in your .env file."
        )

    # Load key content to detect algorithm (RSA vs EC)
    if key_file:
        with open(os.path.expanduser(key_file)) as f:
            key_content = f.read()
    else:
        key_content = key_string

    is_ec = "EC PRIVATE KEY" in key_content
    algorithm = (
        intersight.signing.ALGORITHM_ECDSA_MODE_DETERMINISTIC_RFC6979
        if is_ec else
        intersight.signing.ALGORITHM_RSASSA_PKCS1v15
    )

    signing_kwargs = dict(
        key_id=key_id,
        signing_scheme=intersight.signing.SCHEME_HS2019,
        signing_algorithm=algorithm,
        hash_algorithm=intersight.signing.HASH_SHA256,
        signed_headers=[
            intersight.signing.HEADER_REQUEST_TARGET,
            intersight.signing.HEADER_HOST,
            intersight.signing.HEADER_DATE,
            intersight.signing.HEADER_DIGEST,
        ],
    )
    if key_file:
        signing_kwargs["private_key_path"] = os.path.expanduser(key_file)
    else:
        signing_kwargs["private_key_string"] = key_string

    config = intersight.Configuration(host=base_url)
    config.signing_info = intersight.signing.HttpSigningConfiguration(**signing_kwargs)
    return ApiClient(config)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_environment_summary() -> str:
    """
    Get a high-level summary of the Intersight environment.
    Returns total server count, powered-on count, and active alarm counts
    broken down by severity. Use this for overview questions like
    'how many servers do we have' or 'are there any critical alerts'.
    """
    from intersight.api import compute_api, cond_api

    client = _get_client()
    compute = compute_api.ComputeApi(client)
    alarm_api = cond_api.CondApi(client)

    servers = compute.get_compute_physical_summary_list(inlinecount="allpages", top=1)
    powered_on = compute.get_compute_physical_summary_list(
        filter="OperPowerState eq 'on'", inlinecount="allpages", top=1
    )
    all_alarms = alarm_api.get_cond_alarm_list(
        filter="Acknowledge eq 'None'", inlinecount="allpages", top=1
    )
    critical = alarm_api.get_cond_alarm_list(
        filter="Severity eq 'Critical' and Acknowledge eq 'None'",
        inlinecount="allpages", top=1
    )
    warning = alarm_api.get_cond_alarm_list(
        filter="Severity eq 'Warning' and Acknowledge eq 'None'",
        inlinecount="allpages", top=1
    )

    total = getattr(servers, "count", 0) or 0
    on = getattr(powered_on, "count", 0) or 0

    return (
        f"Intersight Environment Summary\n"
        f"{'─' * 35}\n"
        f"  Total Servers:   {total}\n"
        f"  Powered On:      {on}\n"
        f"  Powered Off:     {total - on}\n"
        f"\n"
        f"  Active Alarms:   {getattr(all_alarms, 'count', 0) or 0}\n"
        f"    Critical:      {getattr(critical, 'count', 0) or 0}\n"
        f"    Warning:       {getattr(warning, 'count', 0) or 0}\n"
    )


@mcp.tool()
def list_servers(limit: int = 25) -> str:
    """
    List compute servers managed by Intersight with their health status,
    hardware model, serial number, and power state.
    Use this when asked which servers exist or to get an overview of the fleet.
    """
    from intersight.api import compute_api

    client = _get_client()
    api = compute_api.ComputeApi(client)
    result = api.get_compute_physical_summary_list(top=limit)

    if not result.results:
        return "No servers found in Intersight."

    lines = [f"Servers ({len(result.results)} shown):\n"]
    for s in result.results:
        alarms = s.alarm_summary or {}
        crit = getattr(alarms, "critical", 0) or 0
        warn = getattr(alarms, "warning", 0) or 0
        alarm_str = f"{crit}C/{warn}W" if (crit or warn) else "healthy"
        lines.append(
            f"  {(s.name or 'Unknown'):<30}"
            f"  {(s.model or 'N/A'):<20}"
            f"  S/N: {(s.serial or 'N/A'):<15}"
            f"  Power: {(s.oper_power_state or 'N/A'):<12}"
            f"  Alarms: {alarm_str}"
        )
    return "\n".join(lines)


@mcp.tool()
def list_alarms(severity: str = "") -> str:
    """
    Get active (unacknowledged) alarms from Intersight.
    Optionally filter by severity: 'Critical', 'Warning', or 'Info'.
    Leave severity empty to get all active alarms.
    Use this when asked about problems, alerts, or health issues.
    """
    from intersight.api import cond_api

    client = _get_client()
    api = cond_api.CondApi(client)

    filter_str = "Acknowledge eq 'None'"
    if severity:
        filter_str += f" and Severity eq '{severity}'"

    result = api.get_cond_alarm_list(filter=filter_str, top=50)

    if not result.results:
        label = f" with severity '{severity}'" if severity else ""
        return f"No active alarms{label} found in Intersight."

    lines = [f"Active Alarms ({len(result.results)} found):\n"]
    for a in result.results:
        lines.append(
            f"  [{a.severity or 'N/A'}] {a.name or 'Unknown'}\n"
            f"    {a.description or 'No description'}\n"
            f"    Affected: {a.affected_mo_display_name or 'N/A'}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def get_server_details(name_or_serial: str) -> str:
    """
    Get detailed information about a specific server identified by its name
    or serial number. Returns CPU, memory, firmware, management IP, and alarm info.
    Use this when asked about a specific server by name or serial.
    """
    from intersight.api import compute_api

    client = _get_client()
    api = compute_api.ComputeApi(client)

    result = api.get_compute_physical_summary_list(
        filter=f"Name eq '{name_or_serial}' or Serial eq '{name_or_serial}'",
        top=1,
    )

    if not result.results:
        return f"No server found in Intersight matching '{name_or_serial}'."

    s = result.results[0]
    alarms = s.alarm_summary or {}

    return (
        f"Server: {s.name}\n"
        f"{'─' * 40}\n"
        f"  Model:           {s.model or 'N/A'}\n"
        f"  Serial:          {s.serial or 'N/A'}\n"
        f"  Management IP:   {s.mgmt_ip_address or 'N/A'}\n"
        f"  Power State:     {s.oper_power_state or 'N/A'}\n"
        f"  CPUs:            {s.num_cpus or 'N/A'}\n"
        f"  Memory:          {s.available_memory or 'N/A'} MB\n"
        f"  Firmware:        {s.firmware or 'N/A'}\n"
        f"  Critical Alarms: {getattr(alarms, 'critical', 0) or 0}\n"
        f"  Warning Alarms:  {getattr(alarms, 'warning', 0) or 0}\n"
    )


@mcp.tool()
def get_firmware_summary() -> str:
    """
    Get a summary of firmware versions running across all servers.
    Returns a count of servers per firmware version.
    Use this when asked about firmware consistency, upgrades, or compliance.
    """
    from intersight.api import compute_api

    client = _get_client()
    api = compute_api.ComputeApi(client)
    result = api.get_compute_physical_summary_list(top=500)

    if not result.results:
        return "No servers found in Intersight."

    versions: dict[str, int] = {}
    for s in result.results:
        fw = s.firmware or "Unknown"
        versions[fw] = versions.get(fw, 0) + 1

    lines = [f"Firmware versions across {len(result.results)} server(s):\n"]
    for fw, count in sorted(versions.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 20)
        lines.append(f"  {fw:<35} {bar} {count}")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
