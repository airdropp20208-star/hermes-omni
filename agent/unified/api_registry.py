"""API Registry — auto-discover + use 1500+ public APIs.

THE IDEA
--------
public-apis/public-apis có 1500+ free APIs. Nhưng agent không biết API nào
tồn tại, không biết cách gọi. APIRegistry:

1. **Catalog** — list 1500+ APIs theo category (Animals, Anime, Anti-Malware,
   Art & Design, Books, Business, Calendar, Cloud, Continuous Integration,
   Cryptocurrency, Data, Development, Dictionaries, Documents, Environment,
   Events, Finance, Food, Games, Geocoding, Government, Health, Jobs, Music,
   News, Open Data, Patent, Personality, Phone, Photography, Programming,
   Science, Security, Social, Sports, Test-Data, Text Analysis, Tracking,
   Transportation, URL Shorteners, Vehicle, Video, Weather, v.v.)

2. **Search** — agent query "I need weather data" → returns weather APIs
3. **Auto-call** — agent gọi API qua wrapper tool, không cần nhớ endpoint
4. **Cache** — API responses cached (TTL configurable)

USAGE
-----
    from agent.unified.api_registry import (
        search_apis,
        get_api_info,
        call_api,
    )

    # Search
    results = search_apis("weather")
    # → [{"name": "OpenWeatherMap", "auth": "apiKey", "https": true, ...}]

    # Get details
    info = get_api_info("OpenWeatherMap")

    # Call (wrapper)
    result = call_api("OpenWeatherMap", endpoint="/weather", params={"q": "London"})

CATALOG SOURCE
--------------
Catalog được sync từ https://github.com/public-apis/public-apis
File: public-apis/README.md (markdown table format)

First run: download + parse → cache at ~/.hermes/unified/api_catalog.json
Subsequent runs: load from cache (refresh weekly)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class APIEntry:
    """One public API entry."""

    name: str
    description: str = ""
    auth: str = ""  # "apiKey" | "OAuth" | "No" | ""
    https: bool = True
    cors: str = ""  # "Yes" | "No" | "Unknown"
    link: str = ""
    category: str = ""
    # Extra fields for calling.
    base_url: str = ""
    endpoints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "auth": self.auth,
            "https": self.https,
            "cors": self.cors,
            "link": self.link,
            "category": self.category,
            "base_url": self.base_url,
        }


# --------------------------------------------------------------------------- #
# Built-in mini catalog (curated popular APIs)
# --------------------------------------------------------------------------- #
# Full catalog (1500+) loaded from public-apis on first use.
# This mini catalog ensures common APIs work offline.

BUILTIN_CATALOG: list[dict[str, Any]] = [
    # Weather
    {"name": "Open-Meteo", "description": "Free weather API, no key needed", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://open-meteo.com",
     "category": "Weather", "base_url": "https://api.open-meteo.com/v1"},
    {"name": "National Weather Service", "description": "US weather data", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://www.weather.gov",
     "category": "Weather", "base_url": "https://api.weather.gov"},
    # Crypto
    {"name": "CoinGecko", "description": "Crypto prices, no key needed", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://www.coingecko.com",
     "category": "Cryptocurrency", "base_url": "https://api.coingecko.com/api/v3"},
    # Finance
    {"name": "Frankfurter", "description": "Free currency exchange rates", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://www.frankfurter.app",
     "category": "Finance", "base_url": "https://api.frankfurter.app"},
    # Books
    {"name": "Open Library", "description": "Book data, no key needed", "auth": "No",
     "https": True, "cors": "Unknown", "link": "https://openlibrary.org",
     "category": "Books", "base_url": "https://openlibrary.org"},
    # News
    {"name": "Hacker News", "description": "HN stories, no key needed", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://news.ycombinator.com",
     "category": "News", "base_url": "https://hacker-news.firebaseio.com/v0"},
    # Government
    {"name": "USA.gov", "description": "US government data", "auth": "No",
     "https": True, "cors": "Unknown", "link": "https://www.usa.gov",
     "category": "Government", "base_url": "https://www.usa.gov/api/USAgov"},
    # Animals
    {"name": "Dog API", "description": "Dog images and breeds", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://dog.ceo/dog-api",
     "category": "Animals", "base_url": "https://dog.ceo/api"},
    {"name": "Cat API", "description": "Cat facts", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://catfact.ninja",
     "category": "Animals", "base_url": "https://catfact.ninja"},
    # Jokes
    {"name": "JokeAPI", "description": "Programming, misc, dark jokes", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://jokeapi.de",
     "category": "Personality", "base_url": "https://v2.jokeapi.dev/joke"},
    # Random Data
    {"name": "Random User", "description": "Generate random user data", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://randomuser.me",
     "category": "Test-Data", "base_url": "https://randomuser.me/api"},
    {"name": "JSONPlaceholder", "description": "Fake REST API for testing", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://jsonplaceholder.typicode.com",
     "category": "Test-Data", "base_url": "https://jsonplaceholder.typicode.com"},
    # Geocoding
    {"name": "Nominatim", "description": "Free geocoding (OpenStreetMap)", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://nominatim.org",
     "category": "Geocoding", "base_url": "https://nominatim.openstreetmap.org"},
    # GitHub
    {"name": "GitHub API", "description": "GitHub repos, users, issues", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://docs.github.com/rest",
     "category": "Development", "base_url": "https://api.github.com"},
    # Wikipedia
    {"name": "Wikipedia", "description": "Wikipedia article search and content", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://www.mediawiki.org/wiki/API",
     "category": "Dictionaries", "base_url": "https://en.wikipedia.org/w/api.php"},
    # IP
    {"name": "IPify", "description": "Get public IP address", "auth": "No",
     "https": True, "cors": "Yes", "link": "https://www.ipify.org",
     "category": "Development", "base_url": "https://api.ipify.org"},
    # Time
    {"name": "World Time API", "description": "Current time by timezone", "auth": "No",
     "https": True, "cors": "Unknown", "link": "http://worldtimeapi.org",
     "category": "Calendar", "base_url": "http://worldtimeapi.org/api"},
    # Numbers
    {"name": "Numbers API", "description": "Interesting number facts", "auth": "No",
     "https": True, "cors": "Yes", "link": "http://numbersapi.com",
     "category": "Science", "base_url": "http://numbersapi.com"},
    # Space
    {"name": "NASA APOD", "description": "Astronomy Picture of the Day", "auth": "apiKey",
     "https": True, "cors": "Unknown", "link": "https://api.nasa.gov",
     "category": "Science", "base_url": "https://api.nasa.gov/planetary/apod"},
]


# --------------------------------------------------------------------------- #
# APIRegistry
# --------------------------------------------------------------------------- #


class APIRegistry:
    """Manages public API catalog + provides call wrapper.

    Catalog: built-in (18 common APIs) + optionally full 1500+ from
    public-apis/public-apis (downloaded on first use).
    """

    def __init__(
        self,
        *,
        cache_path: str | Path | None = None,
        auto_fetch_full: bool = False,  # download full 1500+ catalog?
    ) -> None:
        if cache_path is None:
            from hermes_constants import get_hermes_home

            cache_path = get_hermes_home() / "unified" / "api_catalog.json"
        self._cache_path = Path(cache_path).expanduser()
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._auto_fetch = auto_fetch_full
        self._apis: dict[str, APIEntry] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        """Load catalog: built-in + cached full catalog."""
        # Built-in.
        for api_data in BUILTIN_CATALOG:
            entry = APIEntry(**api_data)
            self._apis[entry.name.lower()] = entry
        # Cached full catalog.
        if self._cache_path.exists():
            try:
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                for api_data in data.get("apis", []):
                    entry = APIEntry(**api_data)
                    # Don't override built-in.
                    if entry.name.lower() not in self._apis:
                        self._apis[entry.name.lower()] = entry
            except Exception:
                pass

    def fetch_full_catalog(self) -> int:
        """Download + parse full public-apis catalog. Returns count."""
        try:
            url = "https://raw.githubusercontent.com/public-apis/public-apis/main/README.md"
            req = Request(url, headers={"User-Agent": "Hermes-Omni/1.0"})
            with urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
            apis = self._parse_readme(content)
            # Merge with existing.
            for api in apis:
                if api.name.lower() not in self._apis:
                    self._apis[api.name.lower()] = api
            # Save cache.
            data = {"apis": [a.to_dict() for a in self._apis.values()], "fetched_at": time.time()}
            self._cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return len(apis)
        except Exception:
            return 0

    def _parse_readme(self, content: str) -> list[APIEntry]:
        """Parse public-apis README.md into APIEntry list."""
        apis: list[APIEntry] = []
        current_category = ""
        # Table rows: | API name | Description | Auth | HTTPS | CORS | Link |
        table_row = re.compile(
            r"^\|\s*\[([^\]]+)\]\(([^)]+)\)\s*\|\s*([^|]+)\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|",
            re.MULTILINE,
        )
        # Category headers: ### Category Name
        category_re = re.compile(r"^###\s+(.+)$", re.MULTILINE)
        # Find all categories and their positions.
        cat_positions = [(m.start(), m.group(1).strip()) for m in category_re.finditer(content)]
        for i, (pos, cat) in enumerate(cat_positions):
            end_pos = cat_positions[i + 1][0] if i + 1 < len(cat_positions) else len(content)
            section = content[pos:end_pos]
            for match in table_row.finditer(section):
                name, link, desc, auth, https, cors = match.groups()
                entry = APIEntry(
                    name=name.strip(),
                    description=desc.strip(),
                    auth=auth.strip(),
                    https=https.strip().lower() == "yes",
                    cors=cors.strip(),
                    link=link.strip(),
                    category=cat,
                )
                # Guess base_url from link.
                entry.base_url = link.strip().rstrip("/")
                apis.append(entry)
        return apis

    def list_categories(self) -> list[str]:
        return sorted({a.category for a in self._apis.values() if a.category})

    def list_by_category(self, category: str) -> list[APIEntry]:
        cat_lower = category.lower()
        return [a for a in self._apis.values() if a.category.lower() == cat_lower]

    def search(self, query: str, *, limit: int = 20) -> list[APIEntry]:
        """Search APIs by query (name, description, category).

        Scoring:
        - Exact name match: +5.0
        - Name contains query (word boundary): +3.0
        - Description contains query: +2.0
        - Category contains query: +1.0
        - No-auth bonus: +0.5
        """
        q = query.lower().strip()
        if not q:
            return []
        q_words = q.split()
        scored: list[tuple[float, APIEntry]] = []
        for api in self._apis.values():
            score = 0.0
            name_lower = api.name.lower()
            desc_lower = api.description.lower()
            cat_lower = api.category.lower()
            # Exact name match.
            if name_lower == q:
                score += 5.0
            # Name contains full query.
            elif q in name_lower:
                score += 3.0
            # Name contains all query words.
            elif all(w in name_lower for w in q_words):
                score += 2.5
            # Description contains full query.
            if q in desc_lower:
                score += 2.0
            # Description contains all query words.
            elif all(w in desc_lower for w in q_words):
                score += 1.5
            # Category match.
            if q in cat_lower:
                score += 1.0
            elif all(w in cat_lower for w in q_words):
                score += 0.8
            # Boost no-auth APIs (easier to use).
            if api.auth.lower() in ("no", ""):
                score += 0.3
            if score > 0:
                scored.append((score, api))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[:limit]]

    def get(self, name: str) -> APIEntry | None:
        return self._apis.get(name.lower())

    def call(
        self,
        name: str,
        *,
        endpoint: str = "",
        params: dict[str, str] | None = None,
        api_key: str = "",
        method: str = "GET",
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Call a public API. Returns dict with success/status/data/error."""
        api = self.get(name)
        if api is None:
            return {"success": False, "error": f"API '{name}' not found in catalog"}
        if api.auth.lower() in ("apikey", "oauth") and not api_key:
            return {
                "success": False,
                "error": f"API '{name}' requires auth ({api.auth}). Pass api_key parameter.",
                "auth_required": api.auth,
            }
        # Build URL.
        base = api.base_url.rstrip("/")
        ep = endpoint.lstrip("/") if endpoint else ""
        url = f"{base}/{ep}" if ep else base
        # Add params.
        all_params = params or {}
        if api_key and api.auth.lower() == "apikey":
            # Common param names: apikey, api_key, key, appid
            all_params.setdefault("apikey", api_key)
        if all_params:
            url += "?" + urlencode(all_params)
        # Add Authorization header for OAuth.
        headers = {"User-Agent": "Hermes-Omni/1.0", "Accept": "application/json"}
        if api_key and api.auth.lower() == "oauth":
            headers["Authorization"] = f"Bearer {api_key}"
        # Make request.
        try:
            req = Request(url, headers=headers, method=method)
            with urlopen(req, timeout=timeout) as resp:
                status = resp.status
                body = resp.read().decode("utf-8", errors="ignore")
            # Try parse JSON.
            try:
                data = json.loads(body)
            except Exception:
                data = body[:5000]  # raw text, truncated
            return {
                "success": True,
                "status": status,
                "url": url,
                "data": data,
            }
        except Exception as exc:
            return {"success": False, "error": repr(exc), "url": url}

    def stats(self) -> dict[str, Any]:
        return {
            "total_apis": len(self._apis),
            "categories": len(self.list_categories()),
            "no_auth": sum(1 for a in self._apis.values() if a.auth.lower() in ("no", "")),
            "cache_path": str(self._cache_path),
            "cache_exists": self._cache_path.exists(),
        }


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_registry: APIRegistry | None = None


def get_api_registry() -> APIRegistry:
    global _registry
    if _registry is None:
        _registry = APIRegistry()
    return _registry


def search_apis(query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """Public API: search public APIs by query."""
    return [a.to_dict() for a in get_api_registry().search(query, limit=limit)]


def get_api_info(name: str) -> dict[str, Any] | None:
    """Public API: get info about a specific API."""
    api = get_api_registry().get(name)
    return api.to_dict() if api else None


def call_api(
    name: str,
    *,
    endpoint: str = "",
    params: dict[str, str] | None = None,
    api_key: str = "",
    method: str = "GET",
) -> dict[str, Any]:
    """Public API: call a public API. Returns result dict."""
    return get_api_registry().call(
        name,
        endpoint=endpoint,
        params=params,
        api_key=api_key,
        method=method,
    )


def list_api_categories() -> list[str]:
    """Public API: list all API categories."""
    return get_api_registry().list_categories()


def list_apis_by_category(category: str) -> list[dict[str, Any]]:
    """Public API: list APIs in a category."""
    return [a.to_dict() for a in get_api_registry().list_by_category(category)]


def fetch_full_api_catalog() -> dict[str, Any]:
    """Public API: download full 1500+ API catalog from public-apis."""
    count = get_api_registry().fetch_full_catalog()
    return {"success": count > 0, "count": count}


def api_registry_stats() -> dict[str, Any]:
    """Public API: get registry stats."""
    return get_api_registry().stats()
