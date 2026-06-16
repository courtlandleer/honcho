#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["honcho-ai", "httpx"]
# ///
"""Load Granola meeting notes into Honcho. (resumable, rate-limit aware)

Uses the Granola MCP server (with OAuth) to fetch meetings and the Honcho Python SDK
to store them. Each meeting becomes a Honcho session. Two-person meetings get full
speaker attribution; multi-person meetings are stored as summaries.

Designed to survive real meeting histories (hundreds of meetings, server rate
limits, interrupted runs):
  - Fetches the FULL meeting history: Granola's list_meetings defaults to the
    last 30 days, so the script windows over a custom date range and detects
    server-side clamping/caps from the response headers
  - OAuth token cached on disk (no browser login every run)
  - Adaptive global throttle + patient backoff (Granola allows ~100 req/min)
  - All fetched data cached on disk; re-runs only fetch what's missing
  - Rate-limited transcripts are marked pending and retried next run,
    never silently downgraded to summary
  - Details fetched via get_meetings in batches of 10 (its documented max)
  - Import state tracked; re-runs skip already-imported meetings; summary-only
    imports get upgraded when a transcript becomes available
  - 'a' at any review prompt = accept defaults for everything remaining

Your own words always reach your peer: "Me" turns are microphone-grounded, so
they're imported and reasoned over in every mode — two-person, multi-person —
except when a meeting looks like an in-person/room-mic recording (multiple
attendees but zero "Them" turns), where "Me" may contain other voices; those
are flagged for review instead of auto-ingested.

External participants are up to you (GRANOLA_EXTERNALS):
    full  - externals become peers; their words/presence build representations (default)
    store - externals become peers and their messages are stored/searchable, but
            Honcho does not derive representations of them (peer observe_me=false)
    none  - no peers are created for externals; only your side and meeting
            summaries are imported (other people exist as names in text)

Identity resolution: drop an aliases.json next to this script to map the
identifiers Granola sees onto canonical peer IDs, so the same person resolves
to ONE peer across meetings and across other integrations (e.g. Gmail):
    { "dan": ["daniel@variant.fund", "dan.b@gmail.com", "Daniel Barabander"] }
Each imported peer is also stamped with the emails/names/sources seen for it
in peer metadata, so other importers can resolve against the same registry.

Environment Variables:
    HONCHO_API_KEY    - Your Honcho API key (get from app.honcho.dev/api-keys)
    GRANOLA_WORKSPACE - Honcho workspace to import into (default: granola)
    GRANOLA_ME_PEER   - Optional peer ID for the note creator (default: derived
                        from your email, e.g. alice@example.com -> alice-example-com)
    GRANOLA_EXTERNALS - full | store | none (default: full), see above
    GRANOLA_EARLIEST  - Start of history window, YYYY-MM-DD (default: 2024-01-01)

Usage:
    uv run honcho_granola.py              # full import (resumable)
    uv run honcho_granola.py --diagnose   # dump tool schemas + listing probes only
    uv run honcho_granola.py --auto       # no prompts, accept defaults
    uv run honcho_granola.py --relist     # redo the full listing sweep (else cached)
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import sys
import threading
import traceback
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx


@dataclass
class Participant:
    name: str
    email: str | None = None
    org: str | None = None


@dataclass
class ParsedParticipants:
    note_creator: Participant | None = None
    others: list[Participant] = field(default_factory=list)


@dataclass
class TranscriptTurn:
    speaker: str
    text: str


# Granola MCP + OAuth endpoints
GRANOLA_MCP_URL = "https://mcp.granola.ai/mcp"
AUTH_BASE = "https://mcp-auth.granola.ai"
OAUTH_REDIRECT_PORT = 8765
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_REDIRECT_PORT}/callback"

# Honcho
# Note: deliberately NOT the HONCHO_WORKSPACE env var — that name is commonly set
# globally for other Honcho tooling, and inheriting it here could dump meetings
# into a primary memory workspace by accident.
HONCHO_WORKSPACE = os.environ.get("GRANOLA_WORKSPACE", "granola")
ME_PEER_OVERRIDE = os.environ.get("GRANOLA_ME_PEER")  # map the note creator to an existing peer ID
EXTERNALS_TIER = os.environ.get("GRANOLA_EXTERNALS", "full")  # full | store | none
HONCHO_BASE_URL = os.environ.get("HONCHO_BASE_URL", "https://api.honcho.dev")
OWNER_EMAIL: str | None = None  # the logged-in Granola account; set from get_account_info
MAX_MESSAGE_LEN = 24000  # Honcho message size limit (25000 max, leave headroom)

# Local state
STATE_DIR = Path(__file__).resolve().parent / ".state"
FETCH_DIR = STATE_DIR / "fetch-cache"
DIAG_DIR = STATE_DIR / "diag"
TOKEN_FILE = STATE_DIR / "token.json"
IMPORTED_FILE = STATE_DIR / "imported.json"
INDEX_FILE = STATE_DIR / "meetings-index.json"

# Throttle / retry
BASE_INTERVAL = 1.5     # seconds between MCP calls when healthy
MAX_INTERVAL = 30.0     # ceiling for adaptive slowdown
MAX_ATTEMPTS = 8        # retries per call on rate limit
MAX_BACKOFF = 120.0     # longest single wait

AUTO_MODE = False       # set by --auto or pressing 'a' during review

RATE_LIMIT_RE = re.compile(r"rate.?limit", re.IGNORECASE)


class RateLimitGiveUp(Exception):
    pass


class TokenExpired(Exception):
    pass


# ---------------------------------------------------------------------------
# Small disk helpers
# ---------------------------------------------------------------------------

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_diag(name: str, content: str) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    (DIAG_DIR / name).write_text(content)


def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        raise SystemExit("\nNo interactive terminal available — use --auto for unattended runs.")


# ---------------------------------------------------------------------------
# OAuth callback handler (must be a class for BaseHTTPRequestHandler)
# ---------------------------------------------------------------------------

class _OAuthCallback(BaseHTTPRequestHandler):
    auth_result: dict[str, str | None] = {"code": None, "error": None}

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        if "code" in params:
            _OAuthCallback.auth_result["code"] = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authenticated! You can close this window.</h1>")
        elif "error" in params:
            _OAuthCallback.auth_result["error"] = params.get("error_description", params["error"])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"<h1>Error: {_OAuthCallback.auth_result['error']}</h1>".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


# ---------------------------------------------------------------------------
# Granola OAuth (with disk-cached token)
# ---------------------------------------------------------------------------

async def _register_client(http_client: httpx.AsyncClient) -> str:
    resp = await http_client.post(
        f"{AUTH_BASE}/oauth2/register",
        json={
            "client_name": "Granola to Honcho Transfer",
            "redirect_uris": [OAUTH_REDIRECT_URI],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Client registration failed: {resp.status_code}")
    return resp.json()["client_id"]


def _store_token_response(client_id: str, tok: dict[str, Any]) -> str:
    expires_in = tok.get("expires_in") or 3600
    save_json(TOKEN_FILE, {
        "client_id": client_id,
        "access_token": tok["access_token"],
        "refresh_token": tok.get("refresh_token"),
        "expires_at": datetime.now(timezone.utc).timestamp() + float(expires_in),
    })
    return tok["access_token"]


async def _browser_auth(http_client: httpx.AsyncClient, client_id: str) -> str:
    _OAuthCallback.auth_result = {"code": None, "error": None}

    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()

    auth_url = f"{AUTH_BASE}/oauth2/authorize?" + urlencode({
        "client_id": client_id,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "response_type": "code",
        "state": "granola-honcho-transfer",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })

    server = HTTPServer(("localhost", OAUTH_REDIRECT_PORT), _OAuthCallback)
    thread = threading.Thread(target=server.handle_request)
    thread.start()

    print("  Opening browser for authentication...")
    webbrowser.open(auth_url)
    thread.join(timeout=120)
    server.server_close()

    auth_result = _OAuthCallback.auth_result
    if auth_result["error"]:
        raise RuntimeError(f"Authentication failed: {auth_result['error']}")
    if not auth_result["code"]:
        raise RuntimeError("Authentication timed out")

    resp = await http_client.post(
        f"{AUTH_BASE}/oauth2/token",
        data={
            "grant_type": "authorization_code",
            "code": auth_result["code"],
            "redirect_uri": OAUTH_REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Token exchange failed: {resp.status_code}")

    print("  Authenticated successfully!")
    return _store_token_response(client_id, resp.json())


async def get_access_token(http_client: httpx.AsyncClient, force_new: bool = False) -> str:
    """Return a valid Granola access token, using disk cache / refresh when possible."""
    cached = load_json(TOKEN_FILE)
    if cached and not force_new:
        if cached.get("expires_at", 0) - 60 > datetime.now(timezone.utc).timestamp():
            return cached["access_token"]
        if cached.get("refresh_token") and cached.get("client_id"):
            resp = await http_client.post(
                f"{AUTH_BASE}/oauth2/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": cached["refresh_token"],
                    "client_id": cached["client_id"],
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code == 200:
                print("  Refreshed Granola token from cache.")
                return _store_token_response(cached["client_id"], resp.json())

    print("\nAuthenticating with Granola...")
    client_id = (cached or {}).get("client_id") or await _register_client(http_client)
    try:
        return await _browser_auth(http_client, client_id)
    except RuntimeError:
        if cached and cached.get("client_id"):
            # cached client may have been purged server-side; register fresh and retry once
            return await _browser_auth(http_client, await _register_client(http_client))
        raise


# ---------------------------------------------------------------------------
# MCP transport with adaptive throttle + rate-limit retries
# ---------------------------------------------------------------------------

class McpClient:
    def __init__(self, http_client: httpx.AsyncClient, access_token: str):
        self.http = http_client
        self.token = access_token
        self.interval = BASE_INTERVAL
        self._last_call = 0.0
        self._reauthed = False

    async def _pace(self):
        now = asyncio.get_event_loop().time()
        wait = self._last_call + self.interval - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = asyncio.get_event_loop().time()

    def _slower(self):
        self.interval = min(self.interval * 1.7, MAX_INTERVAL)

    def _faster(self):
        self.interval = max(BASE_INTERVAL, self.interval * 0.97)

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        resp = await self.http.post(
            GRANOLA_MCP_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        if resp.status_code == 401:
            raise TokenExpired()
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            raise _HttpRateLimited(float(retry_after) if retry_after and retry_after.isdigit() else None)
        if resp.status_code != 200:
            raise RuntimeError(f"MCP call failed: {resp.status_code} - {resp.text[:300]}")

        # SSE response
        if "text/event-stream" in resp.headers.get("content-type", ""):
            result = None
            for line in resp.text.split("\n"):
                if line.strip().startswith("data: "):
                    try:
                        parsed = json.loads(line.strip()[6:])
                        if "result" in parsed:
                            result = parsed
                        elif "error" in parsed:
                            raise RuntimeError(f"MCP error: {parsed['error']}")
                    except json.JSONDecodeError:
                        continue
            if result:
                final = result.get("result", {})
                return final if isinstance(final, dict) else {"result": final}
            raise RuntimeError("No result in SSE response")

        result = resp.json()
        if "error" in result:
            raise RuntimeError(f"MCP error: {result['error']}")
        return result.get("result", {})

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """RPC with pacing, adaptive slowdown, backoff on rate limits, one re-auth."""
        for attempt in range(MAX_ATTEMPTS):
            await self._pace()
            try:
                result = await self._rpc(method, params)
            except TokenExpired:
                if self._reauthed:
                    raise RuntimeError("Granola token expired twice — re-run to re-authenticate.")
                self._reauthed = True
                print("  Granola token expired — re-authenticating...")
                self.token = await get_access_token(self.http, force_new=True)
                continue
            except _HttpRateLimited as rl:
                self._slower()
                wait = rl.retry_after or min(10 * (2 ** attempt), MAX_BACKOFF)
                print(f"   ⚠ HTTP 429 rate limit (attempt {attempt + 1}/{MAX_ATTEMPTS}), waiting {wait:.0f}s "
                      f"(pace now {self.interval:.1f}s/call)...")
                await asyncio.sleep(wait)
                continue
            except (httpx.TimeoutException, httpx.TransportError) as e:
                wait = min(5 * (2 ** attempt), 60)
                print(f"   ⚠ Network error ({e.__class__.__name__}), retrying in {wait:.0f}s...")
                await asyncio.sleep(wait)
                continue

            # Granola often reports rate limits inside the content text, not HTTP status
            text = _peek_text(result)
            if text is not None and len(text) < 600 and RATE_LIMIT_RE.search(text):
                self._slower()
                wait = min(10 * (2 ** attempt), MAX_BACKOFF)
                print(f"   ⚠ Granola rate limit (attempt {attempt + 1}/{MAX_ATTEMPTS}), waiting {wait:.0f}s "
                      f"(pace now {self.interval:.1f}s/call)...")
                await asyncio.sleep(wait)
                continue

            self._faster()
            return result
        raise RateLimitGiveUp(f"still rate limited after {MAX_ATTEMPTS} attempts")

    async def call_tool_text(self, tool: str, arguments: dict[str, Any]) -> str:
        result = await self.call("tools/call", {"name": tool, "arguments": arguments})
        return extract_mcp_text(result)

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self.call("tools/list", {})
        return result.get("tools", [])

    async def account_email(self) -> str | None:
        """The email of the logged-in Granola account (the true 'me')."""
        try:
            text = await self.call_tool_text("get_account_info", {})
            m = re.search(r'"email"\s*:\s*"([^"]+)"', text)
            return m.group(1) if m else None
        except Exception as e:
            print(f"  (could not determine account owner: {e})")
            return None


class _HttpRateLimited(Exception):
    def __init__(self, retry_after: float | None):
        self.retry_after = retry_after


def _peek_text(result: dict[str, Any]) -> str | None:
    content = result.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text")
        return str(text) if text is not None else None
    return None


def extract_mcp_text(result: dict[str, Any]) -> str:
    """Extract text from the first content block of an MCP result."""
    content = result.get("content", [])
    if not isinstance(content, list) or not content:
        raise ValueError(f"MCP response missing content array: {list(result.keys())}")
    first = content[0]
    if not isinstance(first, dict) or "text" not in first:
        raise ValueError(f"MCP content block missing 'text' field: {first}")
    return str(first["text"])


# ---------------------------------------------------------------------------
# Meeting listing — schema-aware pagination
# ---------------------------------------------------------------------------

def parse_meetings_text(text: str) -> list[dict[str, Any]]:
    """Parse Granola's XML-like list_meetings response."""
    meetings: list[dict[str, Any]] = []
    for match in re.finditer(r'<meeting\s+id="([^"]+)"\s+title="([^"]+)"\s+date="([^"]+)"', text):
        mid, title, date = match.groups()
        block_end = text.find("</meeting>", match.end())
        block = text[match.end():block_end] if block_end != -1 else ""
        p_match = re.search(r"<known_participants>\s*(.*?)\s*</known_participants>", block, re.DOTALL)
        meetings.append({
            "id": mid,
            "title": title,
            "date": date,
            "participants": p_match.group(1).strip() if p_match else "",
        })
    return meetings


_TZ_OFFSETS = {"UTC": 0, "GMT": 0, "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
               "MST": -7, "MDT": -6, "PST": -8, "PDT": -7}


def try_parse_date(date_str: str) -> datetime | None:
    s = (date_str or "").strip()
    if not s:
        return None
    offset = 0
    tz_match = re.search(r"\s+([A-Z]{2,4})$", s)
    if tz_match and tz_match.group(1) in _TZ_OFFSETS:
        offset = _TZ_OFFSETS[tz_match.group(1)]
        s = s[:tz_match.start()]
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ["%b %d, %Y %I:%M %p", "%b %d, %Y %I:%M:%S %p", "%B %d, %Y %I:%M %p",
                "%b %d, %Y", "%B %d, %Y"]:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone(timedelta(hours=offset)))
        except ValueError:
            continue
    return None


def parse_date(date_str: str) -> datetime:
    dt = try_parse_date(date_str)
    if dt is None:
        raise ValueError(f"Unrecognized date format: {date_str!r}")
    return dt


def _tool_schema(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for t in tools:
        if t.get("name") == name:
            return t.get("inputSchema") or t.get("input_schema") or {}
    return {}


def _candidate_keys(props: dict[str, Any], needles: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    out = []
    for key in props:
        kl = key.lower()
        if any(n in kl for n in needles) and not any(x in kl for x in exclude):
            out.append(key)
    return out


GRANOLA_EARLIEST = os.environ.get("GRANOLA_EARLIEST", "2024-01-01")
CAP_SUSPECT = 50  # if one response returns this many meetings, assume it may be capped and split


async def _fetch_window(
    mcp: McpClient,
    start: datetime,
    end: datetime,
    seen: dict[str, dict[str, Any]],
    depth: int = 0,
) -> None:
    """Fetch one [start, end] window; split recursively if the response looks capped."""
    args = {
        "time_range": "custom",
        "custom_start": start.strftime("%Y-%m-%d"),
        "custom_end": end.strftime("%Y-%m-%d"),
    }
    text = await mcp.call_tool_text("list_meetings", args)
    if depth == 0:
        save_diag("listing-initial.txt", text)
    batch = parse_meetings_text(text)
    new = [m for m in batch if m["id"] not in seen]
    for m in new:
        seen[m["id"]] = m
    if new:
        print(f"    {args['custom_start']} -> {args['custom_end']}: +{len(new)} meetings (total {len(seen)})")
    else:
        print(f"    {args['custom_start']} -> {args['custom_end']}: no new (verifying coverage)")

    if (end - start).days < 2 or depth >= 14:
        return

    # The server clamps windows (~1 year) and reports what it actually served:
    # <meetings_data from="..." to="..." count="...">. Fetch any unserved early part.
    header = re.search(r'<meetings_data[^>]*\bfrom="([^"]+)"', text)
    served_from = try_parse_date(header.group(1)) if header else None
    if served_from and served_from > start + timedelta(days=2):
        await _fetch_window(mcp, start, served_from + timedelta(days=1), seen, depth + 1)

    # A big batch may also be silently capped — split the window to be sure
    if len(batch) >= CAP_SUSPECT:
        mid = start + (end - start) / 2
        await _fetch_window(mcp, start, mid + timedelta(days=1), seen, depth + 1)
        await _fetch_window(mcp, mid, end, seen, depth + 1)


async def list_all_meetings(mcp: McpClient, tools: list[dict[str, Any]], verbose: bool = False) -> list[dict[str, Any]]:
    """Fetch the complete meeting list.

    Granola's list_meetings defaults to time_range=last_30_days (this is what capped
    v1/v2.0 runs at ~58 meetings). With custom_start/custom_end we window over the
    full history instead, splitting any window that looks capped.
    """
    schema = _tool_schema(tools, "list_meetings")
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if verbose:
        print(f"  list_meetings schema params: {sorted(props.keys()) or '(none documented)'}")

    seen: dict[str, dict[str, Any]] = {}

    if {"custom_start", "custom_end"} <= set(props.keys()):
        end = datetime.now(timezone.utc) + timedelta(days=1)

        # Cached index: the exhaustive sweep is expensive, old meetings don't change —
        # re-runs just refresh the window since the last listing. --relist forces full.
        index = load_json(INDEX_FILE)
        if index and index.get("meetings") and "--relist" not in sys.argv:
            seen = {m["id"]: m for m in index["meetings"]}
            since = datetime.fromisoformat(index["listed_at"]) - timedelta(days=7)
            print(f"  Cached index: {len(seen)} meetings; refreshing {since.date()} -> now "
                  f"(use --relist to redo the full sweep)")
            await _fetch_window(mcp, since, end, seen)
        else:
            start = datetime.strptime(GRANOLA_EARLIEST, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            print(f"  Listing full history {start.date()} -> {end.date()} "
                  f"(custom time_range, override start via GRANOLA_EARLIEST)...")
            await _fetch_window(mcp, start, end, seen)
            oldest = [d for d in (try_parse_date(m.get("date", "")) for m in seen.values()) if d]
            if oldest and min(oldest).date() <= start.date() + timedelta(days=31):
                print(f"  ⚠ Oldest meeting found ({min(oldest).date()}) is near the window start — "
                      f"re-run with GRANOLA_EARLIEST=2022-01-01 if history might go back further.")

        save_json(INDEX_FILE, {
            "listed_at": datetime.now(timezone.utc).isoformat(),
            "meetings": list(seen.values()),
        })
        print(f"  Total meetings: {len(seen)}")
        dated = [(try_parse_date(m.get("date", "")), m) for m in seen.values()]
        dated.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return [m for _, m in dated]

    # Fallback for unknown schemas: single call, then probe generic pagination params
    base_args: dict[str, Any] = {}
    if "limit" in props:
        maximum = props["limit"].get("maximum") if isinstance(props["limit"], dict) else None
        base_args["limit"] = int(maximum) if maximum else 100

    text = await mcp.call_tool_text("list_meetings", base_args)
    save_diag("listing-initial.txt", text)
    for m in parse_meetings_text(text):
        seen.setdefault(m["id"], m)
    print(f"  Initial listing: {len(seen)} meetings")

    strategies_used: list[str] = []

    # Strategy 1: offset-style pagination
    for key in _candidate_keys(props, ("offset", "skip")):
        added = await _paginate_numeric(mcp, base_args, key, seen, mode="offset")
        if added:
            strategies_used.append(f"offset:{key}")
            break

    # Strategy 2: page-number pagination
    if not strategies_used:
        for key in _candidate_keys(props, ("page",), exclude=("size", "token")):
            added = await _paginate_numeric(mcp, base_args, key, seen, mode="page")
            if added:
                strategies_used.append(f"page:{key}")
                break

    # Strategy 3: cursor/token pagination (cursor value scraped from response text)
    if not strategies_used:
        for key in _candidate_keys(props, ("cursor", "token", "next")):
            added = await _paginate_cursor(mcp, base_args, key, seen, text)
            if added:
                strategies_used.append(f"cursor:{key}")
                break

    # Strategy 4: date-window pagination (ask for meetings older than the oldest seen)
    if not strategies_used:
        date_keys = _candidate_keys(props, ("before", "until", "end", "to_", "max_date"))
        date_keys += [k for k in _candidate_keys(props, ("date", "start", "from", "after", "since"))
                      if k not in date_keys]
        for key in date_keys:
            added = await _paginate_date_window(mcp, base_args, key, seen)
            if added:
                strategies_used.append(f"date:{key}")
                break

    label = ", ".join(strategies_used) if strategies_used else "none worked — single page only"
    print(f"  Pagination: {label}. Total meetings: {len(seen)}")
    if not strategies_used:
        print("  ⚠ Could not page past the first batch. Schema + raw response saved to "
              f"{DIAG_DIR}/ — run --diagnose and share the output to debug.")

    dated = [(try_parse_date(m.get("date", "")), m) for m in seen.values()]
    dated.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return [m for _, m in dated]


async def _paginate_numeric(
    mcp: McpClient, base_args: dict[str, Any], key: str, seen: dict[str, dict], mode: str,
) -> bool:
    found_any = False
    value = len(seen) if mode == "offset" else 2
    for _ in range(200):
        args = dict(base_args)
        args[key] = value
        try:
            text = await mcp.call_tool_text("list_meetings", args)
        except Exception as e:
            print(f"    ({mode} probe via {key!r} failed: {e})")
            return found_any
        new = [m for m in parse_meetings_text(text) if m["id"] not in seen]
        if not new:
            return found_any
        found_any = True
        for m in new:
            seen[m["id"]] = m
        print(f"    +{len(new)} meetings via {key}={value} (total {len(seen)})")
        value = len(seen) if mode == "offset" else value + 1
    return found_any


async def _paginate_cursor(
    mcp: McpClient, base_args: dict[str, Any], key: str, seen: dict[str, dict], first_text: str,
) -> bool:
    cursor_re = re.compile(r'(?:next_?cursor|next_?page_?token|cursor)["\'>:\s=]+([A-Za-z0-9+/=_\-]{8,})')
    match = cursor_re.search(first_text)
    if not match:
        return False
    found_any = False
    cursor = match.group(1)
    for _ in range(200):
        args = dict(base_args)
        args[key] = cursor
        try:
            text = await mcp.call_tool_text("list_meetings", args)
        except Exception as e:
            print(f"    (cursor probe via {key!r} failed: {e})")
            return found_any
        new = [m for m in parse_meetings_text(text) if m["id"] not in seen]
        if not new:
            return found_any
        found_any = True
        for m in new:
            seen[m["id"]] = m
        print(f"    +{len(new)} meetings via cursor (total {len(seen)})")
        match = cursor_re.search(text)
        if not match:
            return found_any
        cursor = match.group(1)
    return found_any


async def _paginate_date_window(
    mcp: McpClient, base_args: dict[str, Any], key: str, seen: dict[str, dict],
) -> bool:
    found_any = False
    for _ in range(200):
        dates = [d for d in (try_parse_date(m.get("date", "")) for m in seen.values()) if d]
        if not dates:
            return found_any
        oldest = min(dates) - timedelta(minutes=1)
        args = dict(base_args)
        args[key] = oldest.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            text = await mcp.call_tool_text("list_meetings", args)
        except Exception as e:
            print(f"    (date probe via {key!r} failed: {e})")
            return found_any
        new = [m for m in parse_meetings_text(text) if m["id"] not in seen]
        if not new:
            return found_any
        found_any = True
        for m in new:
            seen[m["id"]] = m
        print(f"    +{len(new)} older meetings via {key}<{args[key]} (total {len(seen)})")
    return found_any


# ---------------------------------------------------------------------------
# Per-meeting fetch with disk cache
# ---------------------------------------------------------------------------

def split_meeting_blocks(text: str) -> dict[str, str]:
    """Split a multi-meeting get_meetings response into per-meeting-ID blocks."""
    blocks: dict[str, str] = {}
    for m in re.finditer(r'<meeting\s+id="([^"]+)"', text):
        end = text.find("</meeting>", m.start())
        blocks[m.group(1)] = text[m.start():end + len("</meeting>")] if end != -1 else text[m.start():]
    return blocks


async def fetch_all(mcp: McpClient, meetings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fetch details + transcripts for all meetings, cached on disk.

    Details come via get_meetings in batches of 10 (its documented max) — far fewer
    API calls than one per meeting. Transcripts must be fetched per meeting.

    transcript_status: 'ok' | 'none' (server has no transcript) | 'rate_limited' | 'error'
    Only 'ok' and 'none' are final; the others retry on the next run.
    """
    records: dict[str, dict[str, Any]] = {}
    for m in meetings:
        cached = load_json(FETCH_DIR / f"{m['id']}.json") or {}
        rec = dict(cached)
        rec.update(m)  # listing fields (title/date/participants) are freshest
        records[m["id"]] = rec

    # Phase 1: batched details for meetings missing them
    need_details = [mid for mid, r in records.items() if not r.get("raw_content")]
    if need_details:
        print(f"\nFetching details for {len(need_details)} meetings "
              f"({(len(need_details) + 9) // 10} batched calls)...")
        for i in range(0, len(need_details), 10):
            chunk = need_details[i:i + 10]
            try:
                text = await mcp.call_tool_text("get_meetings", {"meeting_ids": chunk})
                blocks = split_meeting_blocks(text)
                for mid in chunk:
                    block = blocks.get(mid) or (text if len(chunk) == 1 else "")
                    if block:
                        records[mid]["raw_content"] = block
                        save_json(FETCH_DIR / f"{mid}.json", records[mid])
            except Exception as exc:
                print(f"   Details batch failed ({exc}) — will retry next run")
            print(f"    details {min(i + 10, len(need_details))}/{len(need_details)}")

    # Phase 2: transcripts, one call per meeting
    pending = [m["id"] for m in meetings
               if records[m["id"]].get("transcript_status") not in ("ok", "none")]
    cached_done = len(meetings) - len(pending)
    if cached_done:
        print(f"\n  {cached_done} meetings already fetched (cache) — skipping.")
    if pending:
        print(f"\nFetching transcripts for {len(pending)} meetings...\n")
    counts = {"ok": 0, "none": 0, "rate_limited": 0, "error": 0}
    for m in meetings:
        rec = records[m["id"]]
        if rec.get("transcript_status") in ("ok", "none"):
            counts[rec["transcript_status"]] += 1
            continue
        try:
            text = await mcp.call_tool_text("get_meeting_transcript", {"meeting_id": m["id"]})
            if not text or "no transcript" in text.lower():
                rec["transcript_status"] = "none"
                rec.pop("transcript", None)
            else:
                rec["transcript"] = text
                rec["transcript_status"] = "ok"
        except RateLimitGiveUp:
            rec["transcript_status"] = "rate_limited"
        except Exception as e:
            print(f"   Transcript error: {e}")
            rec["transcript_status"] = "error"
        rec["fetched_at"] = datetime.now(timezone.utc).isoformat()
        save_json(FETCH_DIR / f"{m['id']}.json", rec)

        status = rec["transcript_status"]
        counts[status] += 1
        done = sum(counts.values())
        tag = {"ok": "transcript", "none": "summary only", "rate_limited": "RATE LIMITED — will retry next run",
               "error": "ERROR — will retry next run"}[status]
        print(f"  [{done}/{len(meetings)}] {tag}: {rec.get('title', 'Untitled')[:45]}")

    print(f"\n  Transcripts: {counts['ok']} fetched, {counts['none']} unavailable, "
          f"{counts['rate_limited']} rate-limited, {counts['error']} errored")
    if counts["rate_limited"] or counts["error"]:
        print("  ⚠ Pending transcripts retry automatically on the next run — nothing is lost.")
    return [records[m["id"]] for m in meetings]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def is_calendar_resource(email: str | None, name: str | None) -> bool:
    """Calendar rooms/resources/shared calendars are not people.

    Google exposes meeting rooms as `c_...@resource.calendar.google.com` and
    shared calendars as `...@group.calendar.google.com`; they show up in
    Granola's participant list but must never become peers.
    """
    if email and re.search(r"\.calendar\.google\.com$", email, re.IGNORECASE):
        return True
    return False


def parse_participants(participants_str: str) -> ParsedParticipants:
    """Parse Granola's participant string into structured participants."""
    result = ParsedParticipants()
    if not participants_str:
        return result

    # Split on commas, but not inside angle brackets
    entries, current, depth = [], [], 0
    for ch in participants_str:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            entries.append("".join(current))
            current = []
            continue
        current.append(ch)
    if current:
        entries.append("".join(current))

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        is_creator = "(note creator)" in entry
        clean = entry.replace("(note creator)", "").strip()

        email_match = re.search(r"<([^>]+)>", clean)
        email = email_match.group(1) if email_match else None
        name = re.sub(r"\s*<[^>]+>", "", clean).strip()

        if not name:
            print(f"  Warning: could not parse participant entry: {entry!r}")
            continue

        if is_calendar_resource(email, name):
            continue  # meeting room / shared calendar, not a person

        org = None
        org_match = re.match(r"(.+?)\s+from\s+(.+)", name)
        if org_match:
            name, org = org_match.group(1).strip(), org_match.group(2).strip()

        person = Participant(name=name, email=email, org=org)
        if is_creator:
            result.note_creator = person
        else:
            result.others.append(person)

    return result


def parse_transcript_turns(raw: str) -> list[TranscriptTurn]:
    """Split a Granola transcript into speaker turns."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "transcript" in parsed:
            raw = str(parsed["transcript"])
    except (json.JSONDecodeError, TypeError):
        pass

    parts = re.split(r"(?:^|\s{2,})(Me|Them):\s*", raw)
    turns: list[TranscriptTurn] = []
    i = 1
    while i < len(parts) - 1:
        text = parts[i + 1].strip()
        if text:
            turns.append(TranscriptTurn(speaker=parts[i], text=text))
        i += 2
    return turns


def extract_summary(meeting: dict[str, Any]) -> str:
    """Extract best available summary text from meeting data."""
    candidates = []
    for key in ("summary", "notes", "note", "meeting_notes", "description"):
        val = meeting.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())

    raw = meeting.get("raw_content")
    if isinstance(raw, str) and raw.strip():
        candidates.append(raw.strip())

    for c in candidates:
        for tag in ("summary", "notes"):
            m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", c, re.DOTALL)
            if m:
                return m.group(1).strip()

    return candidates[0] if candidates else ""


def peer_id_from(value: str) -> str:
    """Normalize a name or email into a Honcho-safe peer ID."""
    norm = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower())
    norm = re.sub(r"-{2,}", "-", norm).strip("-_")
    return (norm or "peer")[:100]


# Identity registry: aliases.json maps canonical peer IDs to the identifiers
# (emails, names) a source may report for that person, so the same human
# resolves to one peer across meetings and across integrations.
ALIASES_FILE = Path(__file__).resolve().parent / "aliases.json"


def load_aliases() -> dict[str, str]:
    raw = load_json(ALIASES_FILE, {})
    lookup: dict[str, str] = {}
    for canonical, identifiers in raw.items():
        for ident in identifiers:
            lookup[str(ident).strip().lower()] = canonical
    return lookup


ALIASES = load_aliases()


def resolve_peer_id(p: Participant) -> str:
    for ident in (p.email, p.name):
        if ident and ident.strip().lower() in ALIASES:
            return ALIASES[ident.strip().lower()]
    return peer_id_from(p.email or p.name)


def owner_canonical_id() -> str | None:
    """Canonical peer ID for the logged-in account (honoring aliases/override)."""
    if ME_PEER_OVERRIDE:
        return ME_PEER_OVERRIDE
    if OWNER_EMAIL:
        return resolve_peer_id(Participant(name=OWNER_EMAIL, email=OWNER_EMAIL))
    return None


def creator_is_owner(creator: Participant | None) -> bool:
    """Is the note creator the logged-in account owner?

    Granola notes can be created by a teammate and shared into your account; in
    those the "Me" track is the teammate's microphone, not yours. We only treat a
    meeting as the user's own recording when the creator matches the account owner
    (by email or alias). If the owner is unknown (e.g. offline replay), assume yes
    so single-user imports behave as before.
    """
    if not OWNER_EMAIL:
        return True
    if not creator:
        return False
    if (creator.email or "").strip().lower() == OWNER_EMAIL.strip().lower():
        return True
    oc = owner_canonical_id()
    return oc is not None and resolve_peer_id(creator) == oc


def ensure_peer(honcho: Any, peer_id: str, participant: Participant | None,
                is_external: bool, stamped: set[str]) -> Any:
    """Get/create a peer, stamp its identity metadata, apply the externals tier.

    Metadata stamping records every email/name/source observed for the peer so
    other importers (Gmail, etc.) can resolve identities against the same registry.
    """
    peer = honcho.peer(peer_id)
    if peer_id in stamped:
        return peer
    stamped.add(peer_id)
    try:
        meta = peer.get_metadata() or {}
        changed = False
        for key, value in (("emails", participant.email if participant else None),
                           ("names", participant.name if participant else None),
                           ("sources", "granola")):
            if not value:
                continue
            existing = meta.get(key) or []
            if value not in existing:
                meta[key] = existing + [value]
                changed = True
        if changed:
            peer.set_metadata(meta)
        if is_external and EXTERNALS_TIER == "store":
            from honcho.peer import PeerConfig
            peer.set_configuration(PeerConfig(observe_me=False))
    except Exception as e:
        print(f"  (peer metadata/config for {peer_id} failed: {e})")
    return peer


def sanitize(text: str) -> str:
    """Remove null bytes and control characters."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


# ---------------------------------------------------------------------------
# Honcho import helpers
# ---------------------------------------------------------------------------

def honcho_rest_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ['HONCHO_API_KEY']}"}


def list_existing_session_ids() -> set[str]:
    """List session IDs already in the granola workspace (REST; SDK has no list-by-workspace)."""
    ids: set[str] = set()
    page = 1
    with httpx.Client(timeout=30.0) as client:
        while True:
            resp = client.post(
                f"{HONCHO_BASE_URL}/v3/workspaces/{HONCHO_WORKSPACE}/sessions/list",
                params={"page": page, "size": 100},
                json={},
                headers=honcho_rest_headers(),
            )
            if resp.status_code == 404:
                return ids
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                items, total_pages = data, None
            else:
                items = data.get("items", [])
                total_pages = data.get("pages") or data.get("total_pages")
            for it in items:
                sid = it.get("id") or it.get("name")
                if sid:
                    ids.add(sid)
            if not items or (total_pages and page >= total_pages):
                return ids
            page += 1


def delete_session(session_id: str) -> None:
    with httpx.Client(timeout=30.0) as client:
        resp = client.delete(
            f"{HONCHO_BASE_URL}/v3/workspaces/{HONCHO_WORKSPACE}/sessions/{session_id}",
            headers=honcho_rest_headers(),
        )
        if resp.status_code not in (200, 202, 204, 404):
            resp.raise_for_status()


def build_messages(
    peer: Any,
    content: str,
    metadata: dict[str, object] | None,
    created_at: datetime,
) -> list[Any]:
    """Build chunked messages for a single peer, attaching metadata to the first chunk."""
    messages = []
    content = sanitize(content)
    for start in range(0, len(content), MAX_MESSAGE_LEN):
        chunk = content[start:start + MAX_MESSAGE_LEN]
        msg_meta = metadata if start == 0 else None
        messages.append(peer.message(chunk, metadata=msg_meta, created_at=created_at))
    return messages


def send_messages(session: Any, messages: list[Any]) -> None:
    """Send messages to a session in batches of 100."""
    for batch_start in range(0, len(messages), 100):
        session.add_messages(messages[batch_start:batch_start + 100])


def import_two_person(
    honcho: Any,
    session: Any,
    me_peer_id: str,
    them_peer_id: str,
    turns: list[TranscriptTurn],
    metadata: dict[str, object],
    created_at: datetime,
) -> None:
    """Import a two-person meeting with speaker attribution."""
    me_peer = honcho.peer(me_peer_id)
    them_peer = honcho.peer(them_peer_id)

    merged: list[TranscriptTurn] = []
    for t in turns:
        if merged and merged[-1].speaker == t.speaker:
            merged[-1].text += " " + t.text
        else:
            merged.append(TranscriptTurn(speaker=t.speaker, text=t.text))

    messages: list[Any] = []
    for i, t in enumerate(merged):
        peer = me_peer if t.speaker == "Me" else them_peer
        msg_meta = metadata if i == 0 else None
        messages.extend(build_messages(peer, t.text, msg_meta, created_at))

    send_messages(session, messages)
    print(f"  -> Imported as 2-person ({me_peer_id} + {them_peer_id})")


def import_multi(
    honcho: Any,
    session: Any,
    me_peer_id: str,
    others: list[Participant],
    turns: list[TranscriptTurn],
    meeting: dict[str, Any],
    metadata: dict[str, object],
    created_at: datetime,
    stamped: set[str],
) -> None:
    """Import a multi-person meeting: the user's own (microphone-grounded) turns
    plus the meeting summary. External attendees join the session as members so
    the deriver can observe their presence at summary fidelity — their collapsed
    'Them' speech is never attributed to individuals."""
    me_peer = honcho.peer(me_peer_id)

    summary = extract_summary(meeting) or "No summary available"
    title = meeting.get("title", "Untitled")
    header = (f"Meeting: {title}\nDate: {meeting.get('date', '')}\n"
              f"Participants: {meeting.get('participants', '')}\n\n")
    messages = build_messages(me_peer, header + summary, metadata, created_at)

    my_turns = [t for t in turns if t.speaker == "Me"]
    for t in my_turns:
        messages.extend(build_messages(me_peer, t.text, None, created_at))

    if others and EXTERNALS_TIER != "none":
        try:
            members = [ensure_peer(honcho, resolve_peer_id(p), p, True, stamped) for p in others]
            session.add_peers(members)
        except Exception as e:
            print(f"  (adding session members failed: {e})")

    send_messages(session, messages)
    print(f"  -> Imported as multi ({len(my_turns)} of my turns + summary"
          f"{f', {len(others)} members' if others and EXTERNALS_TIER != 'none' else ''})")


def import_summary(
    honcho: Any,
    session: Any,
    me_peer_id: str,
    meeting: dict[str, Any],
    metadata: dict[str, object],
    created_at: datetime,
) -> None:
    """Import a meeting as a summary message."""
    me_peer = honcho.peer(me_peer_id)
    summary = extract_summary(meeting)
    if not summary:
        raw_t = meeting.get("transcript", "")
        try:
            parsed = json.loads(raw_t)
            summary = str(parsed.get("transcript", "")) if isinstance(parsed, dict) else raw_t
        except (json.JSONDecodeError, TypeError):
            summary = raw_t
    summary = summary or "No content available"

    title = meeting.get("title", "Untitled")
    date = meeting.get("date", "")
    header = f"Meeting: {title}\nDate: {date}\nParticipants: {meeting.get('participants', '')}\n\n"

    messages = build_messages(me_peer, header + summary, metadata, created_at)
    send_messages(session, messages)
    print("  -> Imported as summary")


def resolve_them_participant(others: list[Participant]) -> Participant | None:
    """Ask user to pick which participant is 'Them' from a multi-person meeting."""
    if AUTO_MODE:
        return None
    for j, p in enumerate(others, 1):
        email_str = f" <{p.email}>" if p.email else ""
        print(f"    {j}. {p.name}{email_str}")
    idx_str = ask(f"  Who is 'Them'? [1-{len(others)}]: ").strip()
    try:
        return others[int(idx_str) - 1]
    except (ValueError, IndexError):
        print("  Invalid selection.")
        return None


def review_meeting(
    index: int,
    total: int,
    meeting: dict[str, Any],
    participants: ParsedParticipants,
    turns: list[TranscriptTurn],
) -> tuple[str, Participant | None]:
    """Display meeting info and get user's import choice.

    Returns (mode, them_participant) where mode is one of:
    - "two_person": import with speaker attribution using them_participant
    - "summary": import as a single summary message
    - "skip": skip this meeting (rate-limited transcripts: retried next run)
    """
    global AUTO_MODE
    title = meeting.get("title", "Untitled")
    date = meeting.get("date", "")
    creator = participants.note_creator
    others = participants.others
    t_status = meeting.get("transcript_status", "none")

    me_turns = sum(1 for t in turns if t.speaker == "Me")
    them_turns = len(turns) - me_turns
    total_words = sum(len(t.text.split()) for t in turns)

    has_transcript = bool(meeting.get("transcript"))
    two_person_ok = EXTERNALS_TIER != "none" and len(others) == 1 and them_turns > 0
    # Room-mic suspect: attendees on the invite but only "Me" audio — likely an
    # in-person meeting where the laptop mic captured everyone, so "Me" turns
    # may contain other people's words. Don't auto-ingest those to the user peer.
    room_mic = len(others) >= 1 and me_turns > 0 and them_turns == 0 and has_transcript
    adhoc = them_turns > 0 and not others  # someone spoke, but no attendees listed

    # Foreign note: a teammate created/shared this note, so the "Me" track is
    # THEIR microphone, not yours. Never attribute it to the user's peer.
    if not creator_is_owner(creator):
        who = (creator.email or creator.name) if creator else "someone else"
        if AUTO_MODE:
            return ("skip", None)
        print(f"\n{'─' * 60}")
        print(f"  [{index}/{total}] {title}")
        print(f"  Date: {date}")
        print(f"  ⚑ Created by {who}, not you ({OWNER_EMAIL}). The 'Me' track is")
        print("    their microphone — importing it as yours would corrupt your peer.")
        choice = ask("  [Enter] skip (not your recording) / [s] import summary as your note / [a] auto-rest: ").strip().lower()
        if choice == "a":
            AUTO_MODE = True
            return ("skip", None)
        return ("summary", None) if choice == "s" else ("skip", None)

    # Pending transcript: don't lock in a summary import
    if t_status in ("rate_limited", "error"):
        if AUTO_MODE:
            return ("skip", None)
        print(f"\n{'─' * 60}")
        print(f"  [{index}/{total}] {title}")
        print(f"  Date: {date}")
        print("  ** Transcript not yet fetched (rate limited) — retries next run **")
        choice = ask("  [Enter] skip for now / [s] import summary anyway / [a] auto-rest: ").strip().lower()
        if choice == "a":
            AUTO_MODE = True
            return ("skip", None)
        return ("summary", None) if choice == "s" else ("skip", None)

    if AUTO_MODE:
        if two_person_ok:
            return ("two_person", others[0])
        if room_mic:
            return ("summary", None)
        if me_turns > 0:
            return ("multi", None)
        return ("summary", None)

    print(f"\n{'─' * 60}")
    print(f"  [{index}/{total}] {title}")
    print(f"  Date: {date}")
    if creator:
        print(f"  You:  {creator.name} <{creator.email}>")
    for j, p in enumerate(others, 1):
        email_str = f" <{p.email}>" if p.email else ""
        org_str = f" ({p.org})" if p.org else ""
        print(f"    {j}. {p.name}{email_str}{org_str}")

    if turns:
        print(f"  Transcript: {me_turns} Me, {them_turns} Them, ~{total_words} words")
        if total_words < 30:
            print("  ** Very short — might be empty **")
    elif has_transcript:
        raw = meeting["transcript"]
        print(f"  Transcript: present ({len(raw)} chars) but could not parse speaker turns")
        print(f"  Preview: {raw[:200]!r}")
    else:
        print(f"  Content: {'summary available' if extract_summary(meeting) else 'metadata only'}")

    # Flag: room-mic suspect — safe default is summary only
    if room_mic:
        print("\n  ⚑ Attendees listed but zero 'Them' turns — likely in-person/room-mic,")
        print("    so 'Me' may contain other people's voices. Suggested: summary only.")
        choice = ask("  [Enter] summary only (safe) / [m] also ingest my turns / [k] skip / [a] auto-rest: ").strip().lower()
        while choice not in ("", "m", "k", "a"):
            choice = ask("  [Enter] summary only / [m] ingest my turns / [k] skip / [a] auto-rest: ").strip().lower()
        if choice == "a":
            AUTO_MODE = True
            choice = ""
        if choice == "k":
            return ("skip", None)
        return ("multi", None) if choice == "m" else ("summary", None)

    # Two-person default: exactly one other participant with transcript
    if two_person_ok:
        them_label = others[0].name + (f" <{others[0].email}>" if others[0].email else "")
        print(f"\n  Detected: 2-person call (you + {them_label})")
        choice = ask("  [Enter] 2-person / [m]ulti (my turns + summary) / [s]ummary / [k] skip / [a] auto-rest: ").strip().lower()
        while choice not in ("", "m", "s", "k", "a"):
            choice = ask("  [Enter] 2-person / [m]ulti / [s]ummary / [k] skip / [a] auto-rest: ").strip().lower()
        if choice == "a":
            AUTO_MODE = True
            choice = ""
        if choice == "k":
            return ("skip", None)
        if choice == "s":
            return ("summary", None)
        if choice == "m":
            return ("multi", None)
        return ("two_person", others[0])

    # Multi-person (or ad-hoc) with the user speaking: default = my turns + summary
    if me_turns > 0:
        if adhoc:
            print("\n  ⚑ Ad-hoc meeting: someone spoke but no attendees are listed, so")
            print("    'Them' can't be attributed. Suggested: import my turns + summary.")
        else:
            print(f"\n  {len(others)} participants — 'Them' is collapsed, can't attribute individuals.")
            print("    Suggested: my turns + summary (attendees join the session as members).")
        opts = "  [Enter] multi (my turns + summary)"
        valid = ("", "s", "k", "a")
        if others and EXTERNALS_TIER != "none":
            opts += " / [2] treat as 2-person"
            valid = ("", "2", "s", "k", "a")
        opts += " / [s]ummary only / [k] skip / [a] auto-rest: "
        choice = ask(opts).strip().lower()
        while choice not in valid:
            choice = ask(opts).strip().lower()
        if choice == "a":
            AUTO_MODE = True
            choice = ""
        if choice == "k":
            return ("skip", None)
        if choice == "2":
            them = resolve_them_participant(others)
            if them is None:
                return ("summary", None)
            return ("two_person", them)
        if choice == "s":
            return ("summary", None)
        return ("multi", None)

    # No transcript or the user never spoke
    choice = ask("  [Enter] summary / [k] skip / [a] auto-rest: ").strip().lower()
    while choice not in ("", "k", "a"):
        choice = ask("  [Enter] summary / [k] skip / [a] auto-rest: ").strip().lower()
    if choice == "a":
        AUTO_MODE = True
        choice = ""
    if choice == "k":
        return ("skip", None)
    return ("summary", None)


# ---------------------------------------------------------------------------
# Diagnose mode
# ---------------------------------------------------------------------------

async def diagnose(mcp: McpClient) -> None:
    print("\n--- DIAGNOSE: MCP tool discovery ---")
    tools = await mcp.list_tools()
    save_json(DIAG_DIR / "tools.json", tools)
    for t in tools:
        schema = t.get("inputSchema") or t.get("input_schema") or {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        print(f"\n  {t.get('name')}: {t.get('description', '')[:100]}")
        for pname, pdef in props.items():
            if isinstance(pdef, dict):
                bits = [pdef.get("type", "?")]
                for k in ("default", "maximum", "minimum", "format", "description"):
                    if k in pdef:
                        bits.append(f"{k}={str(pdef[k])[:60]}")
                print(f"      {pname}: {', '.join(bits)}")
            else:
                print(f"      {pname}: {pdef}")

    print("\n--- DIAGNOSE: listing probes ---")
    meetings = await list_all_meetings(mcp, tools, verbose=True)
    dates = [d for d in (try_parse_date(m.get("date", "")) for m in meetings) if d]
    if dates:
        print(f"  Date range: {min(dates).date()} -> {max(dates).date()}")
    print(f"\n  Full schemas + raw listing saved to {DIAG_DIR}/")
    print("  If the count still looks low, inspect those files (or attach them to an issue).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    global AUTO_MODE
    AUTO_MODE = "--auto" in sys.argv
    diag_only = "--diagnose" in sys.argv

    print("=" * 60)
    print("  Granola -> Honcho Meeting Notes Transfer (v2)")
    print("=" * 60)

    if not diag_only and not os.environ.get("HONCHO_API_KEY"):
        print("\nError: HONCHO_API_KEY not set.")
        print("  Get your key at: https://app.honcho.dev/api-keys")
        sys.exit(1)

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        try:
            access_token = await get_access_token(http_client)
            mcp = McpClient(http_client, access_token)

            if diag_only:
                await diagnose(mcp)
                return

            tools = await mcp.list_tools()
            save_json(DIAG_DIR / "tools.json", tools)

            global OWNER_EMAIL
            OWNER_EMAIL = await mcp.account_email()
            if OWNER_EMAIL:
                print(f"  Account owner: {OWNER_EMAIL} "
                      f"(only meetings you recorded import as you)")

            print("\nListing meetings from Granola...")
            meetings = await list_all_meetings(mcp, tools)
            if not meetings:
                print("No meetings found.")
                sys.exit(0)

            records = await fetch_all(mcp, meetings)

            from honcho import Honcho

            honcho = Honcho(workspace_id=HONCHO_WORKSPACE)
            imported: dict[str, Any] = load_json(IMPORTED_FILE, {})
            existing_sessions = list_existing_session_ids()

            # Sessions from a previous (v1) run that this state file doesn't know about
            v1_orphans = {sid for sid in existing_sessions
                          if sid.startswith("meeting-") and sid.removeprefix("meeting-") not in imported}
            replace_orphans = False
            if v1_orphans:
                print(f"\n  Found {len(v1_orphans)} meetings in Honcho from an earlier run "
                      "(no local import state — likely summary-only v1 imports).")
                if AUTO_MODE:
                    replace_orphans = True
                    print("  --auto: replacing them with fresh imports.")
                else:
                    choice = ask("  [Enter] replace with fresh imports / [s] leave untouched: ").strip().lower()
                    replace_orphans = choice != "s"

            stamped: set[str] = set()
            results = {"imported": 0, "skipped": 0, "failed": 0, "already": 0, "pending": 0}

            print("\n" + "=" * 60)
            print("  Review each meeting  ('a' = auto-accept all remaining)")
            print("=" * 60)

            for i, m in enumerate(records, 1):
                mid = m.get("id")
                if not mid:
                    continue
                sid = f"meeting-{mid}"
                t_status = m.get("transcript_status", "none")

                participants = parse_participants(m.get("participants", ""))
                turns = parse_transcript_turns(m["transcript"]) if m.get("transcript") else []
                me_speaks = any(t.speaker == "Me" for t in turns)

                prior = imported.get(mid)
                if prior:
                    # Revisit only imports that can now be upgraded: summaries whose
                    # transcript has since arrived, or pre-v3 summaries that can now
                    # carry the user's own turns (multi mode).
                    upgradable = t_status == "ok" and prior.get("mode") == "summary" and (
                        prior.get("transcript_status") != "ok"
                        or (me_speaks and not prior.get("v3"))
                    )
                    if not upgradable:
                        results["already"] += 1
                        continue
                elif sid in existing_sessions and not replace_orphans:
                    results["already"] += 1
                    continue

                mode, them = review_meeting(i, len(records), m, participants, turns)

                if mode == "skip":
                    if t_status in ("rate_limited", "error"):
                        results["pending"] += 1
                    else:
                        print("  -> Skipped")
                        results["skipped"] += 1
                    continue

                creator = participants.note_creator
                # The "me" peer is always the account owner — never a teammate
                # who created a shared note (review already skipped foreign notes
                # unless the user chose summary, where the note is still THEIRS).
                if creator_is_owner(creator) and creator:
                    me_participant = creator
                elif OWNER_EMAIL:
                    me_participant = Participant(name=None, email=OWNER_EMAIL)
                else:
                    me_participant = creator
                me_peer_id = ME_PEER_OVERRIDE or (
                    resolve_peer_id(me_participant) if me_participant else None)
                if not me_peer_id:
                    print("  -> Skipped (no creator identifier)")
                    results["skipped"] += 1
                    continue

                try:
                    created_at = parse_date(m.get("date", ""))
                    if prior or (sid in existing_sessions and replace_orphans):
                        delete_session(sid)
                        print(f"  (replaced earlier import of {sid})")
                    ensure_peer(honcho, me_peer_id, me_participant, False, stamped)
                    session = honcho.session(sid)
                    metadata: dict[str, object] = {
                        "title": m.get("title", "Untitled"),
                        "date": m.get("date", ""),
                        "granola_meeting_id": mid,
                        "mode": mode,
                        "source": "granola",
                    }
                    try:
                        session.set_metadata(metadata)
                    except Exception as e:
                        print(f"  (session metadata failed: {e})")

                    if mode == "two_person" and them is not None:
                        them_peer_id = resolve_peer_id(them)
                        ensure_peer(honcho, them_peer_id, them, True, stamped)
                        import_two_person(honcho, session, me_peer_id, them_peer_id, turns, metadata, created_at)
                    elif mode == "multi":
                        import_multi(honcho, session, me_peer_id, participants.others, turns,
                                     m, metadata, created_at, stamped)
                    else:
                        import_summary(honcho, session, me_peer_id, m, metadata, created_at)

                    results["imported"] += 1
                    imported[mid] = {
                        "mode": mode,
                        "session": sid,
                        "transcript_status": t_status,
                        "v3": True,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                    save_json(IMPORTED_FILE, imported)

                except ValueError as e:
                    print(f"  -> FAILED: {e}")
                    results["failed"] += 1
                except Exception as e:
                    print(f"  -> FAILED: {e}")
                    traceback.print_exc()
                    results["failed"] += 1

            print("\n" + "=" * 60)
            print("  Transfer Complete!")
            print("=" * 60)
            print(f"\n  Imported:          {results['imported']}")
            print(f"  Already imported:  {results['already']}")
            print(f"  Skipped:           {results['skipped']}")
            print(f"  Pending transcript:{results['pending']}  (re-run to retry these)")
            print(f"  Failed:            {results['failed']}")
            print(f"  Workspace: {HONCHO_WORKSPACE}  (externals tier: {EXTERNALS_TIER})")
            if stamped:
                print(f"  Peers this run: {sorted(stamped)}")

        except KeyboardInterrupt:
            print("\n\nAborted. All fetched data and import state are saved — just re-run to resume.")
            sys.exit(0)
        except Exception as e:
            print(f"\nTransfer failed: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
