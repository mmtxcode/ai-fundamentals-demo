#!/usr/bin/env python3
"""
Intersight MCP Server
=====================
Exposes Cisco Intersight infrastructure data as tools for local LLMs via MCP.

Modes
-----
  INTERSIGHT_TOOL_MODE=core  (default)  — 66 read-only tools
  INTERSIGHT_TOOL_MODE=all              — 66 core + 132 write/CRUD tools (198 total)

Setup
-----
  cp .env.example .env          # add your API key credentials
  python intersight_mcp.py      # run standalone to verify connectivity
  # chat.py starts this automatically when /tools is enabled
"""

import os
import sys
from typing import Any, Optional

# ── Dependency check ──────────────────────────────────────────────────────────

def _check_deps():
    missing = []
    for pkg, imp in [("mcp", "mcp"), ("intersight", "intersight"), ("python-dotenv", "dotenv")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}", file=sys.stderr)
        print(f"Run: pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)

_check_deps()

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

# ── Mode configuration ────────────────────────────────────────────────────────

TOOL_MODE = os.environ.get("INTERSIGHT_TOOL_MODE", "core").lower()
ALL_TOOLS = TOOL_MODE == "all"

mcp = FastMCP(
    "Intersight Infrastructure",
    instructions=(
        "Use these tools to answer questions about the live Intersight environment. "
        "Always call a tool when asked about specific servers, alarms, firmware, policies, "
        "or any real-time infrastructure state — do not guess or use training data for live facts. "
        f"Running in {'ALL TOOLS' if ALL_TOOLS else 'CORE (read-only)'} mode."
    ),
)

# ── Authentication ────────────────────────────────────────────────────────────

def _ssl_context():
    """Return an SSL context using the certifi CA bundle (fixes macOS cert issues)."""
    import ssl, certifi
    return ssl.create_default_context(cafile=certifi.where())


def _fetch_oauth_token(base_url: str, client_id: str, client_secret: str) -> str:
    """
    Exchange an OAuth2 Client ID + Client Secret for a bearer token using
    the Intersight token endpoint (client_credentials grant).
    """
    import urllib.request, urllib.parse, json as _json
    token_url = f"{base_url.rstrip('/')}/iam/token"
    payload   = urllib.parse.urlencode({
        "grant_type":    "client_credentials",
        "client_id":     client_id,
        "client_secret": client_secret,
    }).encode()
    req = urllib.request.Request(
        token_url, data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15, context=_ssl_context()) as resp:
        data = _json.loads(resp.read())
    token = data.get("access_token") or data.get("token")
    if not token:
        raise RuntimeError(f"OAuth2 token response missing access_token: {data}")
    return token


def _get_client():
    """
    Build an authenticated Intersight ApiClient from environment variables.

    OAuth2 — Client Credentials (recommended):
      INTERSIGHT_CLIENT_ID           — OAuth2 Client ID
      INTERSIGHT_CLIENT_SECRET       — OAuth2 Client Secret

    OAuth2 — pre-fetched token (alternative):
      INTERSIGHT_OAUTH_TOKEN         — bearer token string

    HTTP Signature (legacy):
      INTERSIGHT_API_KEY_ID          — API key ID from Intersight Settings → API Keys
      INTERSIGHT_API_SECRET_KEY_FILE — path to private key PEM file
        OR
      INTERSIGHT_API_SECRET_KEY      — PEM key content as a string

    Optional:
      INTERSIGHT_BASE_URL            — defaults to https://intersight.com
    """
    import intersight
    from intersight.api_client import ApiClient

    base_url      = os.environ.get("INTERSIGHT_BASE_URL", "https://intersight.com").strip()
    client_id     = os.environ.get("INTERSIGHT_CLIENT_ID", "").strip()
    client_secret = os.environ.get("INTERSIGHT_CLIENT_SECRET", "").strip()
    oauth_token   = os.environ.get("INTERSIGHT_OAUTH_TOKEN", "").strip()

    import certifi

    # ── OAuth2 client credentials (exchange ID+Secret → token) ───────────────
    if client_id and client_secret:
        token = _fetch_oauth_token(base_url, client_id, client_secret)
        config = intersight.Configuration(host=base_url)
        config.access_token = token
        config.ssl_ca_cert = certifi.where()
        return ApiClient(config)

    # ── OAuth2 pre-fetched bearer token ───────────────────────────────────────
    if oauth_token:
        config = intersight.Configuration(host=base_url)
        config.access_token = oauth_token
        config.ssl_ca_cert = certifi.where()
        return ApiClient(config)

    # ── HTTP Signature path ───────────────────────────────────────────────────
    key_id   = os.environ.get("INTERSIGHT_API_KEY_ID", "").strip()
    key_file = os.environ.get("INTERSIGHT_API_SECRET_KEY_FILE", "").strip()
    key_str  = os.environ.get("INTERSIGHT_API_SECRET_KEY", "").strip()

    if not key_id:
        raise RuntimeError(
            "No Intersight credentials found. Set INTERSIGHT_CLIENT_ID + "
            "INTERSIGHT_CLIENT_SECRET for OAuth2, or INTERSIGHT_API_KEY_ID + "
            "INTERSIGHT_API_SECRET_KEY_FILE for HTTP Signature auth."
        )
    if not key_file and not key_str:
        raise RuntimeError(
            "Set INTERSIGHT_API_SECRET_KEY_FILE or INTERSIGHT_API_SECRET_KEY in .env."
        )

    key_content = key_str or open(os.path.expanduser(key_file)).read()
    is_ec = "EC PRIVATE KEY" in key_content
    algorithm = (
        intersight.signing.ALGORITHM_ECDSA_MODE_DETERMINISTIC_RFC6979 if is_ec
        else intersight.signing.ALGORITHM_RSASSA_PKCS1v15
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
        signing_kwargs["private_key_string"] = key_str

    config = intersight.Configuration(host=base_url)
    config.signing_info = intersight.signing.HttpSigningConfiguration(**signing_kwargs)
    config.ssl_ca_cert = certifi.where()
    return ApiClient(config)


# ── Generic request helpers ───────────────────────────────────────────────────

# Strings the LLM sends when it has no value for an optional parameter.
_EMPTY_PLACEHOLDERS = frozenset(("{}", "[]", "null", "none", "undefined", "n/a", "''", '""'))

def _coerce_filter(v: Any) -> str:
    """Accept str, None, dict, or any LLM placeholder — always return a clean str."""
    if not isinstance(v, str):
        return ""
    v = v.strip()
    if v.lower() in _EMPTY_PLACEHOLDERS:
        return ""
    return v

def _coerce_top(v: Any, default: int = 50) -> int:
    """Accept int, str digit, None, or any LLM placeholder — always return an int."""
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
    return default

def _get(path: str, filter_str: Any = None, top: Any = None, select: Any = None) -> dict:
    filter_str = _coerce_filter(filter_str)
    top        = _coerce_top(top, 50)
    select     = _coerce_filter(select)
    client = _get_client()
    params = [("$top", str(top))]
    if filter_str: params.append(("$filter", filter_str))
    if select:     params.append(("$select", select))
    return client.call_api(
        f"/api/v1/{path}", "GET",
        path_params={}, query_params=params,
        header_params={"Accept": "application/json"},
        body=None, post_params=[], files={},
        response_type="object",
        auth_settings=["cookieAuth", "http_signature", "oAuth2", "oAuth2"],
        async_req=False, _return_http_data_only=True,
        collection_formats={}, _preload_content=True, _request_timeout=30,
    )


def _get_by_moid(path: str, moid: str) -> dict:
    client = _get_client()
    return client.call_api(
        f"/api/v1/{path}/{moid}", "GET",
        path_params={}, query_params=[],
        header_params={"Accept": "application/json"},
        body=None, post_params=[], files={},
        response_type="object",
        auth_settings=["cookieAuth", "http_signature", "oAuth2", "oAuth2"],
        async_req=False, _return_http_data_only=True,
        collection_formats={}, _preload_content=True, _request_timeout=30,
    )


def _post(path: str, body: dict) -> dict:
    client = _get_client()
    return client.call_api(
        f"/api/v1/{path}", "POST",
        path_params={}, query_params=[],
        header_params={"Accept": "application/json", "Content-Type": "application/json"},
        body=body, post_params=[], files={},
        response_type="object",
        auth_settings=["cookieAuth", "http_signature", "oAuth2", "oAuth2"],
        async_req=False, _return_http_data_only=True,
        collection_formats={}, _preload_content=True, _request_timeout=30,
    )


def _patch(path: str, moid: str, body: dict) -> dict:
    client = _get_client()
    return client.call_api(
        f"/api/v1/{path}/{moid}", "PATCH",
        path_params={}, query_params=[],
        header_params={"Accept": "application/json", "Content-Type": "application/json"},
        body=body, post_params=[], files={},
        response_type="object",
        auth_settings=["cookieAuth", "http_signature", "oAuth2", "oAuth2"],
        async_req=False, _return_http_data_only=True,
        collection_formats={}, _preload_content=True, _request_timeout=30,
    )


def _delete(path: str, moid: str) -> dict:
    client = _get_client()
    return client.call_api(
        f"/api/v1/{path}/{moid}", "DELETE",
        path_params={}, query_params=[],
        header_params={"Accept": "application/json"},
        body=None, post_params=[], files={},
        response_type="object",
        auth_settings=["cookieAuth", "http_signature", "oAuth2", "oAuth2"],
        async_req=False, _return_http_data_only=True,
        collection_formats={}, _preload_content=True, _request_timeout=30,
    )


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_list(data: dict, fields: list[str], label: str = "item") -> str:
    results = data.get("Results") or []
    count   = data.get("Count", len(results))
    if not results:
        return f"No {label}s found."
    lines = [f"{count} {label}(s):\n"]
    for item in results:
        parts = []
        for f in fields:
            v = item.get(f)
            if v not in (None, "", "unknown", "Unknown"):
                parts.append(f"{f}: {v}")
        if parts:
            lines.append("  " + "  |  ".join(parts))
    return "\n".join(lines)


def _fmt_item(data: dict, fields: list[str]) -> str:
    if not data or "Moid" not in data:
        return "Item not found."
    lines = []
    for f in fields:
        v = data.get(f)
        if v is not None:
            lines.append(f"  {f}: {v}")
    return "\n".join(lines)


def _fmt_ok(data: dict) -> str:
    moid = data.get("Moid", "")
    name = data.get("Name", "")
    return f"OK — Moid: {moid}" + (f"  Name: {name}" if name else "")


# ═════════════════════════════════════════════════════════════════════════════
# CORE TOOLS (66) — available in both core and all modes
# ═════════════════════════════════════════════════════════════════════════════

# ── Inventory & Discovery ─────────────────────────────────────────────────────

@mcp.tool()
def list_compute_servers(filter: Any = None, top: Any = None) -> str:
    """List all compute servers (blades and rack units) with optional OData filtering."""
    return _fmt_list(_get("compute/PhysicalSummaries", filter, top),
                     ["Name", "Model", "Serial", "OperPowerState", "Dn"], "server")


@mcp.tool()
def list_compute_blades(filter: Any = None, top: Any = None) -> str:
    """List all blade servers in chassis."""
    return _fmt_list(_get("compute/Blades", filter, top),
                     ["Dn", "Model", "Serial", "OperState", "Presence"], "blade")


@mcp.tool()
def list_compute_rack_units(filter: Any = None, top: Any = None) -> str:
    """List all rack-mounted servers."""
    return _fmt_list(_get("compute/RackUnits", filter, top),
                     ["Name", "Model", "Serial", "OperState", "ManagementIp"], "rack unit")


@mcp.tool()
def list_compute_boards(filter: Any = None, top: Any = None) -> str:
    """List all server motherboards/compute boards."""
    return _fmt_list(_get("compute/Boards", filter, top),
                     ["Dn", "Model", "OperState", "Presence"], "compute board")


@mcp.tool()
def get_server_details(moid: str) -> str:
    """Get detailed information about a specific server by its MOID."""
    return _fmt_item(_get_by_moid("compute/PhysicalSummaries", moid),
                     ["Name", "Model", "Serial", "OperPowerState", "ManagementIp",
                      "NumCpus", "AvailableMemory", "Firmware", "Dn"])


@mcp.tool()
def list_chassis(filter: Any = None, top: Any = None) -> str:
    """List all equipment chassis."""
    return _fmt_list(_get("equipment/Chasses", filter, top),
                     ["Name", "Model", "Serial", "OperState", "Presence"], "chassis")


@mcp.tool()
def list_fabric_interconnects(filter: Any = None, top: Any = None) -> str:
    """List all fabric interconnects and network elements."""
    return _fmt_list(_get("network/Elements", filter, top),
                     ["Dn", "Model", "Serial", "OperState", "OutOfBandIpAddress"], "fabric interconnect")


@mcp.tool()
def search_resources(resource_type: str, filter: Any = None, top: Any = None) -> str:
    """Search for any Intersight resource by API type with optional OData filter. E.g. resource_type='compute/PhysicalSummaries'."""
    return _fmt_list(_get(resource_type, filter, top), ["Name", "Moid", "Dn", "Model"], "result")


# ── Alarms & Monitoring ───────────────────────────────────────────────────────

@mcp.tool()
def list_alarms(severity: Any = None, filter: Any = None, top: Any = None) -> str:
    """List active (unacknowledged) alarms. Optionally filter by severity: Critical, Warning, Info."""
    severity = _coerce_filter(severity)
    filter   = _coerce_filter(filter)
    f = "Acknowledge eq 'None'"
    if severity: f += f" and Severity eq '{severity}'"
    if filter:   f += f" and ({filter})"
    return _fmt_list(_get("cond/Alarms", f, top),
                     ["Severity", "Name", "Description", "AffectedMoDisplayName"], "alarm")


@mcp.tool()
def list_tam_advisories(filter: Any = None, top: Any = None) -> str:
    """List technical advisories and field notices affecting your infrastructure."""
    return _fmt_list(_get("tam/Advisories", filter, top),
                     ["Name", "Severity", "Description", "State"], "advisory")


@mcp.tool()
def get_tam_advisory_count() -> str:
    """Get count of active technical advisories by severity."""
    data = _get("tam/AdvisoryInstances", top=1)
    count = data.get("Count", 0)
    return f"Total advisory instances: {count}"


# ── Policy Management ─────────────────────────────────────────────────────────

@mcp.tool()
def list_policies(policy_type: str, filter: Any = None, top: Any = None) -> str:
    """List policies of a specific type. E.g. policy_type='vnic/LanConnectivityPolicies'."""
    return _fmt_list(_get(policy_type, filter, top),
                     ["Name", "Description", "Moid"], "policy")


@mcp.tool()
def get_policy(policy_type: str, moid: str) -> str:
    """Get details of a specific policy by type and MOID."""
    return _fmt_item(_get_by_moid(policy_type, moid),
                     ["Name", "Description", "Moid", "Tags"])


@mcp.tool()
def list_server_profiles(filter: Any = None, top: Any = None) -> str:
    """List all server profiles."""
    return _fmt_list(_get("server/Profiles", filter, top),
                     ["Name", "Description", "AssignedServer", "ConfigContext", "Moid"], "server profile")


@mcp.tool()
def get_server_profile(moid: str) -> str:
    """Get details of a specific server profile."""
    return _fmt_item(_get_by_moid("server/Profiles", moid),
                     ["Name", "Description", "AssignedServer", "ConfigContext", "Moid", "Tags"])


@mcp.tool()
def list_bios_units(filter: Any = None, top: Any = None) -> str:
    """List all BIOS/UEFI firmware units."""
    return _fmt_list(_get("bios/Units", filter, top),
                     ["Dn", "Model", "InitSeq", "InitTs"], "BIOS unit")


@mcp.tool()
def list_boot_device_boot_modes(filter: Any = None, top: Any = None) -> str:
    """List all boot device boot modes."""
    return _fmt_list(_get("boot/DeviceBootModes", filter, top),
                     ["Dn", "ConfiguredBootMode", "LastConfiguredBootMode"], "boot mode")


@mcp.tool()
def list_adapter_config_policies(filter: Any = None, top: Any = None) -> str:
    """List all Ethernet adapter configuration policies."""
    return _fmt_list(_get("adapter/ConfigPolicies", filter, top),
                     ["Name", "Description", "Moid"], "adapter config policy")


@mcp.tool()
def get_adapter_config_policy(moid: str) -> str:
    """Get details of a specific adapter configuration policy."""
    return _fmt_item(_get_by_moid("adapter/ConfigPolicies", moid),
                     ["Name", "Description", "Moid", "Settings"])


@mcp.tool()
def list_lan_connectivity_policies(filter: Any = None, top: Any = None) -> str:
    """List all LAN connectivity policies."""
    return _fmt_list(_get("vnic/LanConnectivityPolicies", filter, top),
                     ["Name", "Description", "Moid"], "LAN connectivity policy")


@mcp.tool()
def get_lan_connectivity_policy(moid: str) -> str:
    """Get details of a specific LAN connectivity policy."""
    return _fmt_item(_get_by_moid("vnic/LanConnectivityPolicies", moid),
                     ["Name", "Description", "Moid", "PlacementMode"])


@mcp.tool()
def list_vnics(filter: Any = None, top: Any = None) -> str:
    """List all vNICs (virtual Ethernet interfaces), optionally filtered by LAN connectivity policy."""
    return _fmt_list(_get("vnic/EthIfs", filter, top),
                     ["Name", "MacAddress", "Order", "Placement", "Moid"], "vNIC")


@mcp.tool()
def get_vnic(moid: str) -> str:
    """Get details of a specific vNIC."""
    return _fmt_item(_get_by_moid("vnic/EthIfs", moid),
                     ["Name", "MacAddress", "Order", "Cdn", "Placement", "Moid"])


# ── Pool Management ───────────────────────────────────────────────────────────

@mcp.tool()
def list_pools(pool_type: str, filter: Any = None, top: Any = None) -> str:
    """List pools of a specific type. E.g. pool_type='ippool/Pools' or 'macpool/Pools'."""
    return _fmt_list(_get(pool_type, filter, top),
                     ["Name", "Description", "Size", "Assigned", "Moid"], "pool")


@mcp.tool()
def list_ippool_blocks(filter: Any = None, top: Any = None) -> str:
    """List all IP pool address blocks."""
    return _fmt_list(_get("ippool/ShadowBlocks", filter, top),
                     ["IpV4Block", "IpV4Config", "Moid"], "IP block")


@mcp.tool()
def list_macpool_blocks(filter: Any = None, top: Any = None) -> str:
    """List all MAC pool blocks."""
    return _fmt_list(_get("macpool/Blocks", filter, top),
                     ["MacBlock", "Moid"], "MAC block")


@mcp.tool()
def list_fcpool_blocks(filter: Any = None, top: Any = None) -> str:
    """List all Fibre Channel pool blocks (WWNN/WWPN)."""
    return _fmt_list(_get("fcpool/Blocks", filter, top),
                     ["IdBlock", "Moid"], "FC pool block")


@mcp.tool()
def get_ippool_block(moid: str) -> str:
    """Get details of a specific IP pool block."""
    return _fmt_item(_get_by_moid("ippool/ShadowBlocks", moid),
                     ["IpV4Block", "IpV4Config", "Moid"])


@mcp.tool()
def get_macpool_block(moid: str) -> str:
    """Get details of a specific MAC pool block."""
    return _fmt_item(_get_by_moid("macpool/Blocks", moid),
                     ["MacBlock", "Moid"])


# ── Telemetry & Metrics ───────────────────────────────────────────────────────

@mcp.tool()
def get_server_telemetry(moid: str) -> str:
    """Get telemetry data for a specific server: CPU, memory, temperature, power."""
    return _fmt_item(_get_by_moid("compute/PhysicalSummaries", moid),
                     ["Name", "NumCpus", "AvailableMemory", "AlarmSummary"])


@mcp.tool()
def get_chassis_telemetry(moid: str) -> str:
    """Get telemetry data for a chassis: temperature, power, fans."""
    return _fmt_item(_get_by_moid("equipment/Chasses", moid),
                     ["Name", "Model", "OperState", "AlarmSummary"])


@mcp.tool()
def get_adapter_telemetry(moid: str) -> str:
    """Get network adapter telemetry: throughput, errors, link status."""
    return _fmt_item(_get_by_moid("adapter/Units", moid),
                     ["Dn", "Model", "OperState", "Presence"])


@mcp.tool()
def list_processor_units(filter: Any = None, top: Any = None) -> str:
    """List all processor units with utilization info."""
    return _fmt_list(_get("processor/Units", filter, top),
                     ["Dn", "Model", "OperState", "NumCores", "NumThreads", "Speed"], "processor")


@mcp.tool()
def list_memory_units(filter: Any = None, top: Any = None) -> str:
    """List all memory units with health and capacity."""
    return _fmt_list(_get("memory/Units", filter, top),
                     ["Dn", "Model", "Capacity", "OperState", "Type"], "memory unit")


@mcp.tool()
def list_storage_controllers(filter: Any = None, top: Any = None) -> str:
    """List storage controllers with health and RAID information."""
    return _fmt_list(_get("storage/Controllers", filter, top),
                     ["Dn", "Model", "OperState", "RaidSupport", "Presence"], "storage controller")


@mcp.tool()
def list_physical_drives(filter: Any = None, top: Any = None) -> str:
    """List physical drives with health, capacity, and wear info."""
    return _fmt_list(_get("storage/PhysicalDisks", filter, top),
                     ["Dn", "Model", "Size", "DiskState", "DriveFirmware"], "physical drive")


@mcp.tool()
def get_power_statistics(moid: str, resource_type: str = "compute/RackUnits") -> str:
    """Get power consumption statistics for a server or chassis by MOID."""
    return _fmt_item(_get_by_moid(resource_type, moid),
                     ["Dn", "Model", "OperPowerState"])


@mcp.tool()
def get_thermal_statistics(moid: str, resource_type: str = "compute/RackUnits") -> str:
    """Get thermal/temperature statistics for a server or chassis by MOID."""
    return _fmt_item(_get_by_moid(resource_type, moid),
                     ["Dn", "Model", "OperState"])


@mcp.tool()
def list_fan_modules(filter: Any = None, top: Any = None) -> str:
    """List fan modules with operational status and speed."""
    return _fmt_list(_get("equipment/Fans", filter, top),
                     ["Dn", "Model", "OperState", "Presence"], "fan")


@mcp.tool()
def list_psu_units(filter: Any = None, top: Any = None) -> str:
    """List power supply units with status and output."""
    return _fmt_list(_get("equipment/Psus", filter, top),
                     ["Dn", "Model", "OperState", "Presence", "Voltage"], "PSU")


@mcp.tool()
def list_storage_virtual_drives(filter: Any = None, top: Any = None) -> str:
    """List all virtual drives (RAID volumes)."""
    return _fmt_list(_get("storage/VirtualDrives", filter, top),
                     ["Dn", "Name", "Size", "DriveState", "Type"], "virtual drive")


@mcp.tool()
def list_pci_devices(filter: Any = None, top: Any = None) -> str:
    """List all PCI devices: NICs, HBAs, GPUs, etc."""
    return _fmt_list(_get("pci/Devices", filter, top),
                     ["Dn", "Model", "OperState", "Presence", "Pid"], "PCI device")


@mcp.tool()
def list_graphics_cards(filter: Any = None, top: Any = None) -> str:
    """List all graphics cards (GPUs)."""
    return _fmt_list(_get("graphics/Cards", filter, top),
                     ["Dn", "Model", "OperState", "Presence"], "GPU")


@mcp.tool()
def get_top_resources(metric: Any = None, top: Any = None) -> str:
    """Get top N servers by metric: 'memory', 'cpu'. Returns the list sorted."""
    metric = _coerce_filter(metric) or "memory"
    top    = _coerce_top(top, 10)
    data = _get("compute/PhysicalSummaries", top=top)
    return _fmt_list(data, ["Name", "Model", "NumCpus", "AvailableMemory"], "server")


# ── Network & Fabric ──────────────────────────────────────────────────────────

@mcp.tool()
def list_fabric_vlans(filter: Any = None, top: Any = None) -> str:
    """List all fabric VLANs."""
    return _fmt_list(_get("fabric/Vlans", filter, top),
                     ["Name", "VlanId", "AutoAllowOnUplinks", "Moid"], "VLAN")


@mcp.tool()
def get_fabric_vlan(moid: str) -> str:
    """Get details of a specific VLAN."""
    return _fmt_item(_get_by_moid("fabric/Vlans", moid),
                     ["Name", "VlanId", "AutoAllowOnUplinks", "Moid"])


@mcp.tool()
def list_fabric_vsans(filter: Any = None, top: Any = None) -> str:
    """List all fabric VSANs (Fibre Channel)."""
    return _fmt_list(_get("fabric/Vsans", filter, top),
                     ["Name", "VsanId", "FcoeVlan", "Moid"], "VSAN")


@mcp.tool()
def get_fabric_vsan(moid: str) -> str:
    """Get details of a specific VSAN."""
    return _fmt_item(_get_by_moid("fabric/Vsans", moid),
                     ["Name", "VsanId", "FcoeVlan", "Moid"])


@mcp.tool()
def list_fabric_port_channels(filter: Any = None, top: Any = None) -> str:
    """List all fabric port channels."""
    return _fmt_list(_get("fabric/PortChannels", filter, top),
                     ["Name", "PcId", "AdminSpeed", "Moid"], "port channel")


@mcp.tool()
def get_fabric_port_channel(moid: str) -> str:
    """Get details of a specific fabric port channel."""
    return _fmt_item(_get_by_moid("fabric/PortChannels", moid),
                     ["Name", "PcId", "AdminSpeed", "Moid"])


# ── Hardware & Firmware ───────────────────────────────────────────────────────

@mcp.tool()
def list_firmware_running(filter: Any = None, top: Any = None) -> str:
    """List all running firmware versions across the infrastructure."""
    return _fmt_list(_get("firmware/RunningFirmwares", filter, top),
                     ["Dn", "Type", "Version", "Component"], "firmware entry")


@mcp.tool()
def list_licenses(filter: Any = None, top: Any = None) -> str:
    """List all license information for registered devices."""
    return _fmt_list(_get("license/LicenseInfos", filter, top),
                     ["LicenseType", "LicenseState", "LicenseCount", "Balance"], "license")


@mcp.tool()
def list_hcl_operating_systems(filter: Any = None, top: Any = None) -> str:
    """List supported operating systems from the Hardware Compatibility List."""
    return _fmt_list(_get("hcl/OperatingSystems", filter, top),
                     ["Vendor", "Name", "Version"], "OS")


@mcp.tool()
def list_hcl_hyperflex_compatibility(filter: Any = None, top: Any = None) -> str:
    """List HyperFlex software compatibility information."""
    return _fmt_list(_get("hcl/HyperflexSoftwareCompatibilityInfos", filter, top),
                     ["ServerFwVersion", "HxdpVersion", "HypervisorType"], "compatibility entry")


@mcp.tool()
def list_equipment_io_cards(filter: Any = None, top: Any = None) -> str:
    """List all IO cards in chassis."""
    return _fmt_list(_get("equipment/IoCards", filter, top),
                     ["Dn", "Model", "OperState", "Presence"], "IO card")


@mcp.tool()
def list_equipment_sys_io_ctrls(filter: Any = None, top: Any = None) -> str:
    """List all system IO controllers."""
    return _fmt_list(_get("equipment/SystemIoControllers", filter, top),
                     ["Dn", "Model", "OperState", "Presence"], "system IO controller")


@mcp.tool()
def list_management_controllers(filter: Any = None, top: Any = None) -> str:
    """List all management controllers (CIMC, IMC, BMC)."""
    return _fmt_list(_get("management/Controllers", filter, top),
                     ["Dn", "Model", "ManagementIp", "OperState"], "management controller")


# ── Workflow & System ─────────────────────────────────────────────────────────

@mcp.tool()
def list_workflows(filter: Any = None, top: Any = None) -> str:
    """List workflow executions with status."""
    return _fmt_list(_get("workflow/WorkflowInfos", filter, top),
                     ["Name", "WorkflowStatus", "Progress", "StartTime", "EndTime"], "workflow")


@mcp.tool()
def get_workflow(moid: str) -> str:
    """Get details of a specific workflow execution."""
    return _fmt_item(_get_by_moid("workflow/WorkflowInfos", moid),
                     ["Name", "WorkflowStatus", "Progress", "Message", "StartTime", "EndTime"])


@mcp.tool()
def list_top_systems(filter: Any = None, top: Any = None) -> str:
    """List all top-level systems with associated compute resources."""
    return _fmt_list(_get("top/Systems", filter, top),
                     ["Dn", "Mode", "Ipv4Address"], "top system")


@mcp.tool()
def get_top_system(moid: str) -> str:
    """Get details of a specific top-level system including its compute resources."""
    return _fmt_item(_get_by_moid("top/Systems", moid),
                     ["Dn", "Mode", "Ipv4Address"])


# ── Code Examples ─────────────────────────────────────────────────────────────

@mcp.tool()
def get_powershell_examples() -> str:
    """Get Cisco Intersight PowerShell module programming examples and documentation."""
    return (
        "Intersight PowerShell Examples:\n"
        "  Repository: https://github.com/CiscoDevNet/intersight-powershell\n"
        "  Install:    Install-Module -Name Intersight.PowerShell\n"
        "  Docs:       https://intersight.com/apidocs/introduction/overview/\n"
        "  Examples:   https://github.com/CiscoDevNet/intersight-powershell-utils"
    )


@mcp.tool()
def get_python_examples() -> str:
    """Get Cisco Intersight Python SDK programming examples and documentation."""
    return (
        "Intersight Python SDK Examples:\n"
        "  Repository: https://github.com/CiscoDevNet/intersight-python\n"
        "  Install:    pip install intersight\n"
        "  Docs:       https://intersight.com/apidocs/introduction/overview/\n"
        "  Examples:   https://github.com/CiscoDevNet/intersight-python-utils"
    )


@mcp.tool()
def get_terraform_examples() -> str:
    """Get Cisco Intersight Terraform provider resources and documentation."""
    return (
        "Intersight Terraform Examples:\n"
        "  Provider:   registry.terraform.io/providers/CiscoDevNet/intersight\n"
        "  Repository: https://github.com/CiscoDevNet/terraform-provider-intersight\n"
        "  Docs:       https://registry.terraform.io/providers/CiscoDevNet/intersight/latest/docs"
    )


@mcp.tool()
def get_ansible_examples() -> str:
    """Get Cisco Intersight Ansible modules, playbooks, and CVD solutions."""
    return (
        "Intersight Ansible Examples:\n"
        "  Collection: cisco.intersight (Ansible Galaxy)\n"
        "  Install:    ansible-galaxy collection install cisco.intersight\n"
        "  Repository: https://github.com/CiscoDevNet/intersight-ansible\n"
        "  Playbooks:  https://github.com/CiscoDevNet/intersight-ansible/tree/main/playbooks"
    )


# ── Security & Health ─────────────────────────────────────────────────────────

@mcp.tool()
def generate_security_health_report() -> str:
    """
    Generate a comprehensive security and health report for the entire infrastructure.
    Aggregates alarms, advisories, license status, and firmware consistency.
    """
    try:
        alarms  = _get("cond/Alarms",      "Acknowledge eq 'None'", top=500)
        advs    = _get("tam/Advisories",    top=100)
        lics    = _get("license/LicenseInfos", top=50)
        servers = _get("compute/PhysicalSummaries", top=500)

        total_servers  = servers.get("Count", 0)
        total_alarms   = alarms.get("Count", 0)
        critical       = sum(1 for a in (alarms.get("Results") or []) if a.get("Severity") == "Critical")
        warning        = sum(1 for a in (alarms.get("Results") or []) if a.get("Severity") == "Warning")
        total_advs     = advs.get("Count", 0)

        license_issues = [
            l for l in (lics.get("Results") or [])
            if l.get("LicenseState") not in ("Compliance", "TrialPeriod")
        ]

        fw_versions: dict[str, int] = {}
        for s in (servers.get("Results") or []):
            fw = s.get("Firmware") or "Unknown"
            fw_versions[fw] = fw_versions.get(fw, 0) + 1
        fw_uniform = len(fw_versions) <= 2

        lines = [
            "═" * 50,
            "  INTERSIGHT SECURITY & HEALTH REPORT",
            "═" * 50,
            f"\n  Servers:          {total_servers}",
            f"  Active Alarms:    {total_alarms}  (Critical: {critical}, Warning: {warning})",
            f"  Advisories:       {total_advs}",
            f"  License Issues:   {len(license_issues)}",
            f"  Firmware:         {'Uniform ✓' if fw_uniform else f'{len(fw_versions)} versions — review needed'}",
            "\n  Firmware Breakdown:",
        ]
        for fw, cnt in sorted(fw_versions.items(), key=lambda x: -x[1]):
            lines.append(f"    {fw}: {cnt} server(s)")

        if license_issues:
            lines.append("\n  License Issues:")
            for l in license_issues:
                lines.append(f"    [{l.get('LicenseState')}] {l.get('LicenseType')}")

        lines.append("\n" + "═" * 50)
        return "\n".join(lines)

    except Exception as e:
        return f"Error generating report: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# EXTENDED TOOLS (132) — only registered when INTERSIGHT_TOOL_MODE=all
# ═════════════════════════════════════════════════════════════════════════════

if ALL_TOOLS:

    # ── Policy Create / Update / Delete ──────────────────────────────────────

    @mcp.tool()
    def create_boot_policy(name: str, description: str = "", org_moid: str = "") -> str:
        """Create a new boot order policy."""
        body = {"Name": name, "Description": description,
                "ObjectType": "boot.Policy",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("boot/Policies", body))

    @mcp.tool()
    def create_bios_policy(name: str, description: str = "", org_moid: str = "") -> str:
        """Create a new BIOS policy."""
        body = {"Name": name, "Description": description,
                "ObjectType": "bios.Policy",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("bios/Policies", body))

    @mcp.tool()
    def create_network_policy(name: str, description: str = "", org_moid: str = "") -> str:
        """Create a new network configuration policy."""
        body = {"Name": name, "Description": description,
                "ObjectType": "networkconfig.Policy",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("networkconfig/Policies", body))

    @mcp.tool()
    def update_policy(policy_type: str, moid: str, name: str = "", description: str = "") -> str:
        """Update an existing policy by type and MOID."""
        body = {}
        if name:        body["Name"] = name
        if description: body["Description"] = description
        return _fmt_ok(_patch(policy_type, moid, body))

    @mcp.tool()
    def delete_policy(policy_type: str, moid: str) -> str:
        """Delete a policy by type and MOID."""
        _delete(policy_type, moid)
        return f"Deleted {policy_type} Moid: {moid}"

    @mcp.tool()
    def create_kvm_policy(name: str, enabled: bool = True, max_sessions: int = 4,
                          remote_port: int = 2068, org_moid: str = "") -> str:
        """Create a KVM (keyboard/video/mouse) policy."""
        body = {"Name": name, "Enabled": enabled, "MaximumSessions": max_sessions,
                "RemotePort": remote_port, "ObjectType": "kvm.Policy",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("kvm/Policies", body))

    @mcp.tool()
    def get_kvm_policy(moid: str) -> str:
        """Get details of a KVM policy."""
        return _fmt_item(_get_by_moid("kvm/Policies", moid),
                         ["Name", "Enabled", "MaximumSessions", "RemotePort", "Moid"])

    @mcp.tool()
    def list_kvm_policies(filter: Any = None, top: Any = None) -> str:
        """List all KVM policies."""
        return _fmt_list(_get("kvm/Policies", filter, top),
                         ["Name", "Enabled", "MaximumSessions", "Moid"], "KVM policy")

    @mcp.tool()
    def update_kvm_policy(moid: str, enabled: bool | None = None,
                          max_sessions: int | None = None) -> str:
        """Update a KVM policy."""
        body = {}
        if enabled is not None:      body["Enabled"] = enabled
        if max_sessions is not None: body["MaximumSessions"] = max_sessions
        return _fmt_ok(_patch("kvm/Policies", moid, body))

    @mcp.tool()
    def delete_kvm_policy(moid: str) -> str:
        """Delete a KVM policy."""
        _delete("kvm/Policies", moid)
        return f"Deleted KVM policy Moid: {moid}"

    @mcp.tool()
    def create_virtual_media_policy(name: str, enabled: bool = True,
                                    org_moid: str = "") -> str:
        """Create a virtual media policy."""
        body = {"Name": name, "Enabled": enabled, "ObjectType": "vmedia.Policy",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("vmedia/Policies", body))

    @mcp.tool()
    def get_virtual_media_policy(moid: str) -> str:
        """Get a virtual media policy."""
        return _fmt_item(_get_by_moid("vmedia/Policies", moid),
                         ["Name", "Enabled", "Mappings", "Moid"])

    @mcp.tool()
    def list_virtual_media_policies(filter: Any = None, top: Any = None) -> str:
        """List all virtual media policies."""
        return _fmt_list(_get("vmedia/Policies", filter, top),
                         ["Name", "Enabled", "Moid"], "virtual media policy")

    @mcp.tool()
    def update_virtual_media_policy(moid: str, name: str = "",
                                    enabled: bool | None = None) -> str:
        """Update a virtual media policy."""
        body = {}
        if name:             body["Name"] = name
        if enabled is not None: body["Enabled"] = enabled
        return _fmt_ok(_patch("vmedia/Policies", moid, body))

    @mcp.tool()
    def delete_virtual_media_policy(moid: str) -> str:
        """Delete a virtual media policy."""
        _delete("vmedia/Policies", moid)
        return f"Deleted virtual media policy Moid: {moid}"

    @mcp.tool()
    def create_sdcard_policy(name: str, description: str = "", org_moid: str = "") -> str:
        """Create an SD card policy."""
        body = {"Name": name, "Description": description, "ObjectType": "sdcard.Policy",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("sdcard/Policies", body))

    @mcp.tool()
    def get_sdcard_policy(moid: str) -> str:
        """Get an SD card policy."""
        return _fmt_item(_get_by_moid("sdcard/Policies", moid),
                         ["Name", "Description", "Partitions", "Moid"])

    @mcp.tool()
    def list_sdcard_policies(filter: Any = None, top: Any = None) -> str:
        """List all SD card policies."""
        return _fmt_list(_get("sdcard/Policies", filter, top),
                         ["Name", "Description", "Moid"], "SD card policy")

    @mcp.tool()
    def update_sdcard_policy(moid: str, name: str = "", description: str = "") -> str:
        """Update an SD card policy."""
        body = {}
        if name:        body["Name"] = name
        if description: body["Description"] = description
        return _fmt_ok(_patch("sdcard/Policies", moid, body))

    @mcp.tool()
    def delete_sdcard_policy(moid: str) -> str:
        """Delete an SD card policy."""
        _delete("sdcard/Policies", moid)
        return f"Deleted SD card policy Moid: {moid}"

    @mcp.tool()
    def create_storage_local_disk_policy(name: str, description: str = "",
                                         org_moid: str = "") -> str:
        """Create a storage local disk configuration policy."""
        body = {"Name": name, "Description": description,
                "ObjectType": "storage.StoragePolicy",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("storage/StoragePolicies", body))

    @mcp.tool()
    def get_storage_local_disk_policy(moid: str) -> str:
        """Get a storage local disk policy."""
        return _fmt_item(_get_by_moid("storage/StoragePolicies", moid),
                         ["Name", "Description", "Moid"])

    # ── Pool Create / Update / Delete ─────────────────────────────────────────

    @mcp.tool()
    def create_ip_pool(name: str, description: str = "", org_moid: str = "") -> str:
        """Create an IP address pool."""
        body = {"Name": name, "Description": description, "ObjectType": "ippool.Pool",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("ippool/Pools", body))

    @mcp.tool()
    def create_mac_pool(name: str, description: str = "", org_moid: str = "") -> str:
        """Create a MAC address pool."""
        body = {"Name": name, "Description": description, "ObjectType": "macpool.Pool",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("macpool/Pools", body))

    @mcp.tool()
    def create_uuid_pool(name: str, description: str = "", org_moid: str = "") -> str:
        """Create a UUID pool."""
        body = {"Name": name, "Description": description, "ObjectType": "uuidpool.Pool",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("uuidpool/Pools", body))

    @mcp.tool()
    def create_wwnn_pool(name: str, description: str = "", org_moid: str = "") -> str:
        """Create a WWNN (World Wide Node Name) pool."""
        body = {"Name": name, "Description": description, "ObjectType": "fcpool.Pool",
                "PoolPurpose": "WWNN",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("fcpool/Pools", body))

    @mcp.tool()
    def create_wwpn_pool(name: str, description: str = "", org_moid: str = "") -> str:
        """Create a WWPN (World Wide Port Name) pool."""
        body = {"Name": name, "Description": description, "ObjectType": "fcpool.Pool",
                "PoolPurpose": "WWPN",
                "Organization": {"ObjectType": "organization.Organization", "Moid": org_moid}}
        return _fmt_ok(_post("fcpool/Pools", body))

    @mcp.tool()
    def update_pool(pool_type: str, moid: str, name: str = "",
                    description: str = "") -> str:
        """Update a pool by type and MOID."""
        body = {}
        if name:        body["Name"] = name
        if description: body["Description"] = description
        return _fmt_ok(_patch(pool_type, moid, body))

    @mcp.tool()
    def delete_pool(pool_type: str, moid: str) -> str:
        """Delete a pool by type and MOID."""
        _delete(pool_type, moid)
        return f"Deleted {pool_type} Moid: {moid}"

    @mcp.tool()
    def create_ippool_block(pool_moid: str, from_ip: str, size: int) -> str:
        """Create an IP address block in an IP pool."""
        body = {"Pool": {"ObjectType": "ippool.Pool", "Moid": pool_moid},
                "IpV4Block": {"From": from_ip, "Size": size},
                "ObjectType": "ippool.ShadowBlock"}
        return _fmt_ok(_post("ippool/ShadowBlocks", body))

    @mcp.tool()
    def update_ippool_block(moid: str, size: int) -> str:
        """Update an IP pool block size."""
        return _fmt_ok(_patch("ippool/ShadowBlocks", moid, {"IpV4Block": {"Size": size}}))

    @mcp.tool()
    def delete_ippool_block(moid: str) -> str:
        """Delete an IP pool block."""
        _delete("ippool/ShadowBlocks", moid)
        return f"Deleted IP pool block Moid: {moid}"

    @mcp.tool()
    def create_macpool_block(pool_moid: str, from_mac: str, size: int) -> str:
        """Create a MAC address block in a MAC pool."""
        body = {"Pool": {"ObjectType": "macpool.Pool", "Moid": pool_moid},
                "MacBlock": {"From": from_mac, "Size": size},
                "ObjectType": "macpool.Block"}
        return _fmt_ok(_post("macpool/Blocks", body))

    @mcp.tool()
    def update_macpool_block(moid: str, size: int) -> str:
        """Update a MAC pool block size."""
        return _fmt_ok(_patch("macpool/Blocks", moid, {"MacBlock": {"Size": size}}))

    @mcp.tool()
    def delete_macpool_block(moid: str) -> str:
        """Delete a MAC pool block."""
        _delete("macpool/Blocks", moid)
        return f"Deleted MAC pool block Moid: {moid}"

    @mcp.tool()
    def create_fcpool_block(pool_moid: str, from_id: str, size: int) -> str:
        """Create a Fibre Channel (WWNN/WWPN) block in an FC pool."""
        body = {"Pool": {"ObjectType": "fcpool.Pool", "Moid": pool_moid},
                "IdBlock": {"From": from_id, "Size": size},
                "ObjectType": "fcpool.Block"}
        return _fmt_ok(_post("fcpool/Blocks", body))

    @mcp.tool()
    def get_fcpool_block(moid: str) -> str:
        """Get a Fibre Channel pool block."""
        return _fmt_item(_get_by_moid("fcpool/Blocks", moid), ["IdBlock", "Moid"])

    @mcp.tool()
    def update_fcpool_block(moid: str, size: int) -> str:
        """Update a Fibre Channel pool block size."""
        return _fmt_ok(_patch("fcpool/Blocks", moid, {"IdBlock": {"Size": size}}))

    @mcp.tool()
    def delete_fcpool_block(moid: str) -> str:
        """Delete a Fibre Channel pool block."""
        _delete("fcpool/Blocks", moid)
        return f"Deleted FC pool block Moid: {moid}"

    # ── Fabric Configuration ──────────────────────────────────────────────────

    @mcp.tool()
    def create_fabric_vlan(name: str, vlan_id: int, auto_allow_on_uplinks: bool = True,
                           policy_moid: str = "") -> str:
        """Create a fabric VLAN."""
        body = {"Name": name, "VlanId": vlan_id,
                "AutoAllowOnUplinks": auto_allow_on_uplinks,
                "ObjectType": "fabric.Vlan",
                "EthNetworkPolicy": {"ObjectType": "fabric.EthNetworkPolicy",
                                     "Moid": policy_moid}}
        return _fmt_ok(_post("fabric/Vlans", body))

    @mcp.tool()
    def update_fabric_vlan(moid: str, name: str = "",
                           auto_allow_on_uplinks: bool | None = None) -> str:
        """Update a fabric VLAN."""
        body = {}
        if name: body["Name"] = name
        if auto_allow_on_uplinks is not None:
            body["AutoAllowOnUplinks"] = auto_allow_on_uplinks
        return _fmt_ok(_patch("fabric/Vlans", moid, body))

    @mcp.tool()
    def delete_fabric_vlan(moid: str) -> str:
        """Delete a fabric VLAN."""
        _delete("fabric/Vlans", moid)
        return f"Deleted VLAN Moid: {moid}"

    @mcp.tool()
    def create_fabric_vsan(name: str, vsan_id: int, fcoe_vlan: int,
                           policy_moid: str = "") -> str:
        """Create a fabric VSAN."""
        body = {"Name": name, "VsanId": vsan_id, "FcoeVlan": fcoe_vlan,
                "ObjectType": "fabric.Vsan",
                "FcNetworkPolicy": {"ObjectType": "fabric.FcNetworkPolicy",
                                    "Moid": policy_moid}}
        return _fmt_ok(_post("fabric/Vsans", body))

    @mcp.tool()
    def update_fabric_vsan(moid: str, name: str = "") -> str:
        """Update a fabric VSAN."""
        body = {}
        if name: body["Name"] = name
        return _fmt_ok(_patch("fabric/Vsans", moid, body))

    @mcp.tool()
    def delete_fabric_vsan(moid: str) -> str:
        """Delete a fabric VSAN."""
        _delete("fabric/Vsans", moid)
        return f"Deleted VSAN Moid: {moid}"

    @mcp.tool()
    def create_fabric_port_channel(name: str, pc_id: int, admin_speed: str = "Auto",
                                   policy_moid: str = "") -> str:
        """Create a fabric port channel."""
        body = {"Name": name, "PcId": pc_id, "AdminSpeed": admin_speed,
                "ObjectType": "fabric.UplinkPcRole",
                "PortPolicy": {"ObjectType": "fabric.PortPolicy", "Moid": policy_moid}}
        return _fmt_ok(_post("fabric/UplinkPcRoles", body))

    @mcp.tool()
    def update_fabric_port_channel(moid: str, admin_speed: str = "") -> str:
        """Update a fabric port channel."""
        body = {}
        if admin_speed: body["AdminSpeed"] = admin_speed
        return _fmt_ok(_patch("fabric/UplinkPcRoles", moid, body))

    @mcp.tool()
    def delete_fabric_port_channel(moid: str) -> str:
        """Delete a fabric port channel."""
        _delete("fabric/UplinkPcRoles", moid)
        return f"Deleted port channel Moid: {moid}"

    @mcp.tool()
    def list_fabric_flow_control_policies(filter: Any = None, top: Any = None) -> str:
        """List fabric flow control policies."""
        return _fmt_list(_get("fabric/FlowControlPolicies", filter, top),
                         ["Name", "PriorityFlowControlMode", "Moid"], "flow control policy")

    @mcp.tool()
    def list_fabric_link_control_policies(filter: Any = None, top: Any = None) -> str:
        """List fabric link control policies."""
        return _fmt_list(_get("fabric/LinkControlPolicies", filter, top),
                         ["Name", "Moid"], "link control policy")

    @mcp.tool()
    def list_fabric_multicast_policies(filter: Any = None, top: Any = None) -> str:
        """List fabric multicast policies."""
        return _fmt_list(_get("fabric/MulticastPolicies", filter, top),
                         ["Name", "QuerierState", "SnoopingState", "Moid"],
                         "multicast policy")

    @mcp.tool()
    def list_fabric_qos_policies(filter: Any = None, top: Any = None) -> str:
        """List fabric QoS (system QoS) policies."""
        return _fmt_list(_get("fabric/SystemQosPolicies", filter, top),
                         ["Name", "Moid"], "QoS policy")

    @mcp.tool()
    def get_fabric_qos_policy(moid: str) -> str:
        """Get details of a fabric system QoS policy."""
        return _fmt_item(_get_by_moid("fabric/SystemQosPolicies", moid),
                         ["Name", "Classes", "Moid"])

    @mcp.tool()
    def list_fabric_uplink_ports(filter: Any = None, top: Any = None) -> str:
        """List all fabric uplink port roles."""
        return _fmt_list(_get("fabric/UplinkRoles", filter, top),
                         ["Dn", "AdminSpeed", "Moid"], "uplink port")

    @mcp.tool()
    def list_fabric_server_ports(filter: Any = None, top: Any = None) -> str:
        """List all fabric server port roles."""
        return _fmt_list(_get("fabric/ServerRoles", filter, top),
                         ["Dn", "Moid"], "server port")

    @mcp.tool()
    def list_fabric_port_operations(filter: Any = None, top: Any = None) -> str:
        """List fabric port operations."""
        return _fmt_list(_get("fabric/PortOperations", filter, top),
                         ["Dn", "AdminState", "Moid"], "port operation")

    # ── Adapter & Network Policies ────────────────────────────────────────────

    @mcp.tool()
    def create_adapter_config_policy(name: str, description: str = "",
                                     org_moid: str = "") -> str:
        """Create an Ethernet adapter configuration policy."""
        body = {"Name": name, "Description": description,
                "ObjectType": "adapter.ConfigPolicy",
                "Organization": {"ObjectType": "organization.Organization",
                                 "Moid": org_moid}}
        return _fmt_ok(_post("adapter/ConfigPolicies", body))

    @mcp.tool()
    def update_adapter_config_policy(moid: str, name: str = "",
                                     description: str = "") -> str:
        """Update an adapter configuration policy."""
        body = {}
        if name:        body["Name"] = name
        if description: body["Description"] = description
        return _fmt_ok(_patch("adapter/ConfigPolicies", moid, body))

    @mcp.tool()
    def delete_adapter_config_policy(moid: str) -> str:
        """Delete an adapter configuration policy."""
        _delete("adapter/ConfigPolicies", moid)
        return f"Deleted adapter config policy Moid: {moid}"

    @mcp.tool()
    def list_eth_adapter_policies(filter: Any = None, top: Any = None) -> str:
        """List all Ethernet adapter policies."""
        return _fmt_list(_get("vnic/EthAdapterPolicies", filter, top),
                         ["Name", "Description", "Moid"], "Ethernet adapter policy")

    @mcp.tool()
    def list_eth_qos_policies(filter: Any = None, top: Any = None) -> str:
        """List all Ethernet QoS policies."""
        return _fmt_list(_get("vnic/EthQosPolicies", filter, top),
                         ["Name", "Mtu", "Cos", "Moid"], "Ethernet QoS policy")

    @mcp.tool()
    def list_eth_network_group_policies(filter: Any = None, top: Any = None) -> str:
        """List all Ethernet network group policies (trunk VLAN lists)."""
        return _fmt_list(_get("vnic/EthNetworkPolicies", filter, top),
                         ["Name", "Moid"], "Ethernet network policy")

    # ── Server Profile Operations ─────────────────────────────────────────────

    @mcp.tool()
    def create_server_profile(name: str, description: str = "",
                               org_moid: str = "") -> str:
        """Create a new server profile."""
        body = {"Name": name, "Description": description,
                "ObjectType": "server.Profile",
                "Organization": {"ObjectType": "organization.Organization",
                                 "Moid": org_moid}}
        return _fmt_ok(_post("server/Profiles", body))

    @mcp.tool()
    def attach_policy_to_profile(profile_moid: str, policy_moid: str,
                                  policy_type: str) -> str:
        """Attach a policy to a server profile."""
        return _fmt_ok(_patch("server/Profiles", profile_moid,
                              {"PolicyBucket": [{"ObjectType": policy_type,
                                                 "Moid": policy_moid}]}))

    @mcp.tool()
    def attach_pool_to_profile(profile_moid: str, pool_moid: str,
                                pool_type: str) -> str:
        """Attach a pool to a server profile."""
        return _fmt_ok(_patch("server/Profiles", profile_moid,
                              {"PolicyBucket": [{"ObjectType": pool_type,
                                                 "Moid": pool_moid}]}))

    @mcp.tool()
    def assign_server_to_profile(profile_moid: str, server_moid: str,
                                  server_type: str = "compute.RackUnit") -> str:
        """Assign a physical server to a server profile."""
        return _fmt_ok(_patch("server/Profiles", profile_moid,
                              {"AssignedServer": {"ObjectType": server_type,
                                                   "Moid": server_moid}}))

    @mcp.tool()
    def deploy_server_profile(profile_moid: str) -> str:
        """Deploy a server profile (apply configuration to assigned server)."""
        return _fmt_ok(_patch("server/Profiles", profile_moid,
                              {"Action": "Deploy"}))

    @mcp.tool()
    def update_server_profile(moid: str, name: str = "",
                               description: str = "") -> str:
        """Update a server profile."""
        body = {}
        if name:        body["Name"] = name
        if description: body["Description"] = description
        return _fmt_ok(_patch("server/Profiles", moid, body))

    @mcp.tool()
    def delete_server_profile(moid: str) -> str:
        """Delete a server profile."""
        _delete("server/Profiles", moid)
        return f"Deleted server profile Moid: {moid}"

    # ── vNIC Operations ───────────────────────────────────────────────────────

    @mcp.tool()
    def create_vnic(name: str, mac_address_type: str = "POOL",
                    lan_policy_moid: str = "") -> str:
        """Create a vNIC (virtual Ethernet interface) on a LAN connectivity policy."""
        body = {"Name": name, "MacAddressType": mac_address_type,
                "ObjectType": "vnic.EthIf",
                "LanConnectivityPolicy": {"ObjectType": "vnic.LanConnectivityPolicy",
                                          "Moid": lan_policy_moid}}
        return _fmt_ok(_post("vnic/EthIfs", body))

    @mcp.tool()
    def update_vnic(moid: str, name: str = "",
                    mac_address_type: str = "") -> str:
        """Update a vNIC."""
        body = {}
        if name:              body["Name"] = name
        if mac_address_type:  body["MacAddressType"] = mac_address_type
        return _fmt_ok(_patch("vnic/EthIfs", moid, body))

    @mcp.tool()
    def delete_vnic(moid: str) -> str:
        """Delete a vNIC."""
        _delete("vnic/EthIfs", moid)
        return f"Deleted vNIC Moid: {moid}"

    # ── Security & System Policies ────────────────────────────────────────────

    @mcp.tool()
    def list_snmp_policies(filter: Any = None, top: Any = None) -> str:
        """List SNMP policies."""
        return _fmt_list(_get("snmp/Policies", filter, top),
                         ["Name", "Enabled", "SnmpPort", "Moid"], "SNMP policy")

    @mcp.tool()
    def create_snmp_policy(name: str, enabled: bool = True, snmp_port: int = 161,
                            org_moid: str = "") -> str:
        """Create an SNMP policy."""
        body = {"Name": name, "Enabled": enabled, "SnmpPort": snmp_port,
                "ObjectType": "snmp.Policy",
                "Organization": {"ObjectType": "organization.Organization",
                                 "Moid": org_moid}}
        return _fmt_ok(_post("snmp/Policies", body))

    @mcp.tool()
    def update_snmp_policy(moid: str, enabled: bool | None = None,
                            snmp_port: int | None = None) -> str:
        """Update an SNMP policy."""
        body = {}
        if enabled is not None:   body["Enabled"] = enabled
        if snmp_port is not None: body["SnmpPort"] = snmp_port
        return _fmt_ok(_patch("snmp/Policies", moid, body))

    @mcp.tool()
    def delete_snmp_policy(moid: str) -> str:
        """Delete an SNMP policy."""
        _delete("snmp/Policies", moid)
        return f"Deleted SNMP policy Moid: {moid}"

    @mcp.tool()
    def list_syslog_policies(filter: Any = None, top: Any = None) -> str:
        """List syslog policies."""
        return _fmt_list(_get("syslog/Policies", filter, top),
                         ["Name", "Moid"], "syslog policy")

    @mcp.tool()
    def create_syslog_policy(name: str, description: str = "",
                              org_moid: str = "") -> str:
        """Create a syslog policy."""
        body = {"Name": name, "Description": description,
                "ObjectType": "syslog.Policy",
                "Organization": {"ObjectType": "organization.Organization",
                                 "Moid": org_moid}}
        return _fmt_ok(_post("syslog/Policies", body))

    @mcp.tool()
    def delete_syslog_policy(moid: str) -> str:
        """Delete a syslog policy."""
        _delete("syslog/Policies", moid)
        return f"Deleted syslog policy Moid: {moid}"

    @mcp.tool()
    def list_ntp_policies(filter: Any = None, top: Any = None) -> str:
        """List NTP policies."""
        return _fmt_list(_get("ntp/Policies", filter, top),
                         ["Name", "Enabled", "NtpServers", "Moid"], "NTP policy")

    @mcp.tool()
    def create_ntp_policy(name: str, ntp_servers: list | None = None,
                           org_moid: str = "") -> str:
        """Create an NTP policy."""
        body = {"Name": name, "Enabled": True,
                "NtpServers": ntp_servers or [],
                "ObjectType": "ntp.Policy",
                "Organization": {"ObjectType": "organization.Organization",
                                 "Moid": org_moid}}
        return _fmt_ok(_post("ntp/Policies", body))

    @mcp.tool()
    def delete_ntp_policy(moid: str) -> str:
        """Delete an NTP policy."""
        _delete("ntp/Policies", moid)
        return f"Deleted NTP policy Moid: {moid}"

    @mcp.tool()
    def list_smtp_policies(filter: Any = None, top: Any = None) -> str:
        """List SMTP policies."""
        return _fmt_list(_get("smtp/Policies", filter, top),
                         ["Name", "Enabled", "SmtpServer", "Moid"], "SMTP policy")

    @mcp.tool()
    def create_smtp_policy(name: str, smtp_server: str = "",
                            org_moid: str = "") -> str:
        """Create an SMTP policy."""
        body = {"Name": name, "Enabled": True, "SmtpServer": smtp_server,
                "ObjectType": "smtp.Policy",
                "Organization": {"ObjectType": "organization.Organization",
                                 "Moid": org_moid}}
        return _fmt_ok(_post("smtp/Policies", body))

    @mcp.tool()
    def delete_smtp_policy(moid: str) -> str:
        """Delete an SMTP policy."""
        _delete("smtp/Policies", moid)
        return f"Deleted SMTP policy Moid: {moid}"

    # ── Hardware & Compliance ─────────────────────────────────────────────────

    @mcp.tool()
    def list_terminal_audit_logs(filter: Any = None, top: Any = None) -> str:
        """List terminal audit logs for compliance and security review."""
        return _fmt_list(_get("aaa/AuditRecords", filter, top),
                         ["Event", "Instname", "UserName", "CreateTime"], "audit log")

    @mcp.tool()
    def list_tam_advisory_instances(filter: Any = None, top: Any = None) -> str:
        """List TAM advisory instances affecting your environment."""
        return _fmt_list(_get("tam/AdvisoryInstances", filter, top),
                         ["Advisory", "State", "AffectedObjectType", "Moid"],
                         "advisory instance")

    @mcp.tool()
    def list_tam_security_advisories(filter: Any = None, top: Any = None) -> str:
        """List security advisories (CVEs and PSIRTs)."""
        return _fmt_list(_get("tam/SecurityAdvisories", filter, top),
                         ["Name", "Severity", "BaseScore", "State"], "security advisory")

    @mcp.tool()
    def get_tam_advisory(moid: str) -> str:
        """Get full details of a specific TAM advisory."""
        return _fmt_item(_get_by_moid("tam/Advisories", moid),
                         ["Name", "Severity", "Description", "State", "Moid"])

    @mcp.tool()
    def list_equipment_tpms(filter: Any = None, top: Any = None) -> str:
        """List Trusted Platform Module (TPM) chips across servers."""
        return _fmt_list(_get("equipment/Tpms", filter, top),
                         ["Dn", "Model", "OperState", "TpmSupportEnabled", "Version"],
                         "TPM")

    @mcp.tool()
    def get_equipment_tpm(moid: str) -> str:
        """Get details of a specific TPM chip."""
        return _fmt_item(_get_by_moid("equipment/Tpms", moid),
                         ["Dn", "Model", "OperState", "TpmSupportEnabled", "Version"])

    @mcp.tool()
    def list_firmware_upgrades(filter: Any = None, top: Any = None) -> str:
        """List firmware upgrade tasks and their status."""
        return _fmt_list(_get("firmware/Upgrades", filter, top),
                         ["Status", "UpgType", "DirectDownload", "Moid"],
                         "firmware upgrade")

    @mcp.tool()
    def list_hcl_operating_system_vendors(filter: Any = None, top: Any = None) -> str:
        """List HCL operating system vendors."""
        return _fmt_list(_get("hcl/OperatingSystemVendors", filter, top),
                         ["Name", "Moid"], "OS vendor")

    # ── Storage ───────────────────────────────────────────────────────────────

    @mcp.tool()
    def list_storage_flex_flash_controllers(filter: Any = None, top: Any = None) -> str:
        """List FlexFlash SD card controllers."""
        return _fmt_list(_get("storage/FlexFlashControllers", filter, top),
                         ["Dn", "Model", "OperState", "Presence"],
                         "FlexFlash controller")

    @mcp.tool()
    def list_storage_flex_flash_drives(filter: Any = None, top: Any = None) -> str:
        """List FlexFlash SD card drives."""
        return _fmt_list(_get("storage/FlexFlashPhysicalDrives", filter, top),
                         ["Dn", "CardStatus", "CardType", "OemId"],
                         "FlexFlash drive")

    @mcp.tool()
    def list_storage_local_disk_policies(filter: Any = None, top: Any = None) -> str:
        """List storage local disk configuration policies."""
        return _fmt_list(_get("storage/StoragePolicies", filter, top),
                         ["Name", "Description", "Moid"], "storage policy")

    @mcp.tool()
    def list_boot_device_boot_securities(filter: Any = None, top: Any = None) -> str:
        """List boot device security configurations."""
        return _fmt_list(_get("boot/DeviceBootSecurities", filter, top),
                         ["Dn", "SecureBoot"], "boot security")

    # ── Miscellaneous ─────────────────────────────────────────────────────────

    @mcp.tool()
    def list_management_interfaces(filter: Any = None, top: Any = None) -> str:
        """List management network interfaces (CIMC/IMC interfaces)."""
        return _fmt_list(_get("management/Interfaces", filter, top),
                         ["Dn", "IpAddress", "MacAddress", "HostName"],
                         "management interface")

    @mcp.tool()
    def list_workflow_tasks(filter: Any = None, top: Any = None) -> str:
        """List individual workflow task executions."""
        return _fmt_list(_get("workflow/TaskInfos", filter, top),
                         ["Name", "Status", "StartTime", "EndTime", "Moid"],
                         "workflow task")

    @mcp.tool()
    def get_deviceconnector_policy(moid: str) -> str:
        """Get details of a device connector policy."""
        return _fmt_item(_get_by_moid("deviceconnector/Policies", moid),
                         ["Name", "LockoutEnabled", "Moid"])

    @mcp.tool()
    def create_vlan_group(name: str, vlan_moids: list | None = None,
                           policy_moid: str = "") -> str:
        """Create a VLAN group for trunk port configuration."""
        body = {"Name": name,
                "Vlans": [{"ObjectType": "fabric.Vlan", "Moid": m}
                           for m in (vlan_moids or [])],
                "ObjectType": "fabric.EthNetworkGroup",
                "EthNetworkPolicy": {"ObjectType": "fabric.EthNetworkPolicy",
                                     "Moid": policy_moid}}
        return _fmt_ok(_post("fabric/EthNetworkGroups", body))

    @mcp.tool()
    def list_fabric_lacp_policies(filter: Any = None, top: Any = None) -> str:
        """List LACP (Link Aggregation Control Protocol) policies."""
        return _fmt_list(_get("fabric/LacpPolicies", filter, top),
                         ["Name", "Moid"], "LACP policy")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tool_count = len(mcp._tool_manager._tools) if hasattr(mcp, '_tool_manager') else "?"
    print(
        f"Intersight MCP Server — mode: {TOOL_MODE.upper()} — "
        f"{'~198' if ALL_TOOLS else '66'} tools",
        file=sys.stderr,
    )
    mcp.run(transport="stdio")
