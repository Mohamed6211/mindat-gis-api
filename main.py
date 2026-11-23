import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mindat GIS API")

# Mindat API v1 (https://api.mindat.org/v1/...). MINDAT_API_URL can be overridden for tests.
MINDAT_API_URL = os.getenv("MINDAT_API_URL", "https://api.mindat.org/v1").rstrip("/")
MINDAT_API_KEY = os.getenv("MINDAT_API_KEY")

# Add your GIS application's domain in the Render environment, e.g.
#   ALLOWED_ORIGINS=https://my-gis-app.example.com,http://localhost:8080
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Mindat sits behind Cloudflare, which can refuse non-browser user agents.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
session = requests.Session()


# ---------------------------------------------------------------------------
# Small in-memory cache (answers are shared by all users of the service)
# ---------------------------------------------------------------------------
class TTLCache:
    def __init__(self, ttl, maxsize=2000):
        self.ttl, self.maxsize = ttl, maxsize
        self._data = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            if item[0] < time.time():
                del self._data[key]
                return None
            return item[1]

    def set(self, key, value):
        with self._lock:
            if len(self._data) >= self.maxsize:          # drop the oldest entries
                for k in sorted(self._data, key=lambda k: self._data[k][0])[: self.maxsize // 10]:
                    del self._data[k]
            self._data[key] = (time.time() + self.ttl, value)


near_cache = TTLCache(6 * 3600)
country_cache = TTLCache(6 * 3600)
MINERAL_NAMES = {}                                       # mineral id -> name (these never change)
_names_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Mindat requests
# ---------------------------------------------------------------------------
def _parse_json(resp):
    try:
        return resp.json()
    except ValueError:
        try:                                              # Mindat sometimes sends raw control characters inside text fields
            return json.loads(re.sub(r"[\x00-\x1f]", " ", resp.text))
        except ValueError:
            raise HTTPException(status_code=502, detail="Mindat returned invalid JSON")


def mindat_request(endpoint, params=None, method="GET", body=None, timeout=30, retries=2):
    if not MINDAT_API_KEY:
        raise HTTPException(status_code=500, detail="Mindat API key is not configured")
    headers = {
        "Authorization": f"Token {MINDAT_API_KEY}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    url = f"{MINDAT_API_URL}/{endpoint.strip('/')}/"
    for attempt in range(retries + 1):
        try:
            resp = session.request(method, url, params=params, json=body, headers=headers, timeout=timeout)
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="Mindat API request timed out")
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Mindat API unreachable ({e.__class__.__name__})")
        status = resp.status_code
        if status == 429 or status >= 500:                # busy: wait a little and try again
            if attempt < retries:
                retry_after = resp.headers.get("Retry-After", "")
                time.sleep(min(float(retry_after), 5.0) if retry_after.replace(".", "", 1).isdigit() else 1.0 + attempt)
                continue
            raise HTTPException(status_code=502, detail=f"Mindat API request failed (upstream status: {status})")
        if status in (401, 403):
            raise HTTPException(status_code=502, detail="Mindat refused the API key (check MINDAT_API_KEY)")
        if status == 404:
            raise HTTPException(status_code=404, detail="Not found on Mindat")
        if status >= 400:
            raise HTTPException(status_code=502, detail=f"Mindat API request failed (upstream status: {status})")
        return _parse_json(resp)
    raise HTTPException(status_code=502, detail="Mindat API request failed")


def _results(data):
    """Items of a Mindat answer: paginated {'results': [...]} or a plain list."""
    if isinstance(data, dict):
        data = data.get("results", [])
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Routes of the first version (same URLs; they now use the Mindat v1 endpoints)
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {"status": "online", "service": "Mindat GIS API"}


@app.get("/health")
def health():
    return {"status": "ok", "key_configured": bool(MINDAT_API_KEY)}


# 1. Search minerals by name
@app.get("/api/minerals/search")
def search_minerals(
    q: str = Query(..., min_length=2, max_length=100),
    size: int = Query(12, ge=1, le=50),
):
    return mindat_request("geomaterials-search", {"q": q, "size": size})


# 2. Retrieve mineral information
@app.get("/api/minerals")
def get_minerals(
    name: Optional[str] = None,
    ima: Optional[bool] = None,
    page_size: int = Query(100, ge=1, le=500),
):
    params = {"fields": "id,name", "page-size": page_size}
    if name:
        params["q"] = name
    if ima is not None:
        params["ima"] = int(ima)
    return mindat_request("geomaterials", params)


# 4. Search localities
@app.get("/api/localities/search")
def search_localities(
    txt: Optional[str] = Query(None, max_length=150),
    country: Optional[str] = Query(None, max_length=100),
    page_size: int = Query(100, ge=1, le=500),
):
    params = {"page-size": page_size, "fields": "id,txt,country,latitude,longitude"}
    if txt:
        params["txt"] = txt
    if country:
        params["country"] = country
    return mindat_request("localities", params)


# ---------------------------------------------------------------------------
# NEW: minerals near a point (click on the map)
#   1 request to Mindat per probe (geomin-point) + 1 request per 100 unknown minerals (names)
#   instead of ~165 requests made by the browser.
# ---------------------------------------------------------------------------
def hex_grid(lat, lon, radius_km):
    """Probe points covering a large radius: Mindat returns only ~10 localities per probe, so
    a hexagonal grid of probes is used (spacing 35-90 km, probe distance = spacing)."""
    s = min(90.0, max(35.0, radius_km / 4.5))
    coslat = max(0.05, abs(math.cos(math.radians(lat))))
    n = math.ceil(radius_km / s) + 1
    pts = []
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * 0.866
            if math.hypot(x, y) <= radius_km:
                pts.append((lat + y / 111.0, lon + x / (111.0 * coslat), s))
    if not pts:
        pts.append((lat, lon, max(1.0, radius_km)))
    return pts[:96]


def probes_for(lat, lon, radius_km):
    if radius_km <= 40:                                    # small radius: one probe at the exact distance
        return [(lat, lon, radius_km)]
    return hex_grid(lat, lon, radius_km)


def _geomin_probe(probe):
    plat, plon, dist = probe
    data = mindat_request(
        "geomin-point",
        method="POST",
        body={"point": {"lat": plat, "lon": plon}, "distance": f"{max(1, int(round(dist)))}km"},
        timeout=40,
    )
    out = {}
    if isinstance(data, dict):                             # { "<mineral_id>": [locality_id, ...], ... }
        for k, v in data.items():
            try:
                mid = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, list):
                out[mid] = {int(x) for x in v if str(x).strip().isdigit()}
    return out


def resolve_mineral_names(ids):
    """id -> name, 100 ids per request, names already known are not asked again."""
    with _names_lock:
        need = [i for i in ids if i not in MINERAL_NAMES]
    chunks = [need[k:k + 100] for k in range(0, len(need), 100)]

    def fetch(chunk):
        data = mindat_request(
            "geomaterials",
            {"id_in": ",".join(str(i) for i in chunk), "fields": "id,name", "page-size": len(chunk)},
        )
        found = {}
        for m in _results(data):
            if isinstance(m, dict) and m.get("id") is not None and m.get("name"):
                found[int(m["id"])] = m["name"]
        return found

    if chunks:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for found in pool.map(fetch, chunks):
                with _names_lock:
                    MINERAL_NAMES.update(found)
    with _names_lock:
        return {i: MINERAL_NAMES.get(i) for i in ids}


@app.get("/api/minerals/near")
def minerals_near(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(5.0, ge=0.1, le=300),
    max_minerals: int = Query(150, ge=1, le=500),
):
    key = (round(lat, 3), round(lon, 3), round(radius_km, 1), max_minerals)     # nearby clicks share one answer
    cached = near_cache.get(key)
    if cached is not None:
        return cached
    probes = probes_for(lat, lon, radius_km)
    merged = {}                                             # mineral id -> set of locality ids
    errors, first_error = 0, None
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_geomin_probe, p) for p in probes]
        for f in futures:
            try:
                for mid, locs in f.result().items():
                    merged.setdefault(mid, set()).update(locs)
            except HTTPException as e:                      # one failed probe does not spoil the others
                errors += 1
                first_error = first_error or e
    if errors == len(probes):                              # nothing came back at all: say why, do not cache
        raise first_error or HTTPException(status_code=502, detail="Mindat did not answer")
    ids = sorted(merged, key=lambda m: (-len(merged[m]), m))
    truncated = len(ids) > max_minerals
    ids = ids[:max_minerals]
    names = resolve_mineral_names(ids)
    minerals = [{"id": i, "name": names[i], "localities": len(merged[i])} for i in ids if names.get(i)]
    result = {
        "lat": lat, "lon": lon, "radius_km": radius_km, "probes": len(probes),
        "minerals": minerals, "names": [m["name"] for m in minerals], "truncated": truncated,
    }
    near_cache.set(key, result)
    return result


# ---------------------------------------------------------------------------
# NEW: localities of a country where a mineral has been reported
#   mineral name -> ids (geomaterials), ids -> locality ids (loc-by-min),
#   locality ids -> localities of the country (localities?id_in=...&country=...), by chunks in parallel.
# ---------------------------------------------------------------------------
def resolve_mineral_ids(name, limit=25):
    """Mineral name -> [(id, name)]. An exact name match wins (one id); otherwise at most 5 partial matches."""
    wanted = name.strip().lower()
    partial = []
    for extra in ({"name": name}, {"q": name}, {"name": f"*{name}*"}):
        try:
            data = mindat_request("geomaterials", dict(extra, fields="id,name", **{"page-size": limit}))
        except HTTPException:
            continue
        found = [m for m in _results(data) if isinstance(m, dict) and m.get("id") is not None]
        exact = [m for m in found if str(m.get("name", "")).strip().lower() == wanted]
        if exact:
            return [(int(m["id"]), m.get("name")) for m in exact[:1]]
        if not partial:
            partial = [m for m in found if wanted in str(m.get("name", "")).lower()]
    return [(int(m["id"]), m.get("name")) for m in partial[:5]]


def _localities_of_mineral(mineral_id):
    """Locality ids where ONE mineral has been reported (loc-by-min)."""
    for exc in ("0", None):                                 # the endpoint wants a non-empty "exc"
        params = {"inc": str(mineral_id)}
        if exc:
            params["exc"] = exc
        try:
            data = mindat_request("loc-by-min", params, timeout=60)
        except HTTPException:
            continue
        items = data if isinstance(data, list) else _results(data)
        return [int(x) for x in items if str(x).strip().isdigit()]
    return []


def localities_with_minerals(ids):
    """Union of the localities of each mineral. NOT one request with all ids in "inc": Mindat reads that
    list as "localities that have ALL these minerals", which gives nothing as soon as there are 2+ different minerals."""
    found = set()
    ids = list(ids)[:5]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for part in pool.map(_localities_of_mineral, ids):
            found.update(part)
    return sorted(found)


@app.get("/api/minerals/in-country")
def minerals_in_country(
    mineral: str = Query(..., min_length=2, max_length=100),
    country: str = Query(..., min_length=2, max_length=100),
    max_localities: int = Query(600, ge=1, le=3000),
):
    key = (mineral.strip().lower(), country.strip().lower(), max_localities)
    cached = country_cache.get(key)
    if cached is not None:
        return cached
    started = time.monotonic()
    resolved = resolve_mineral_ids(mineral.strip())
    base = {"mineral": mineral, "country": country, "mineral_ids": [m[0] for m in resolved], "mineral_names": [m[1] for m in resolved]}
    if not resolved:
        return dict(base, count=0, total_with_mineral=0, truncated=False, localities=[])
    all_ids = localities_with_minerals(base["mineral_ids"])
    chunk, max_chunks = 100, 40                              # at most 4000 localities are looked at
    chunks = [all_ids[k:k + chunk] for k in range(0, len(all_ids), chunk)][:max_chunks]
    truncated = len(all_ids) > chunk * max_chunks

    def fetch(ids_chunk):
        if time.monotonic() - started > 80:                  # keep inside the platform time limit
            return None
        params = {"id_in": ",".join(str(i) for i in ids_chunk), "country": country,
                  "fields": "id,txt,latitude,longitude,country", "page-size": len(ids_chunk)}
        try:
            data = mindat_request("localities", params)
        except HTTPException:
            params.pop("fields")                             # (in case a field name is refused)
            try:
                data = mindat_request("localities", params)
            except HTTPException:
                return []
        return [
            {"id": loc.get("id"), "txt": loc.get("txt"), "latitude": loc.get("latitude"),
             "longitude": loc.get("longitude"), "country": loc.get("country") or country}
            for loc in _results(data) if isinstance(loc, dict)
        ]

    localities = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for part in pool.map(fetch, chunks):
            if part is None:
                truncated = True
                continue
            localities.extend(part)
    localities = localities[:max_localities]
    result = dict(base, count=len(localities), total_with_mineral=len(all_ids), truncated=truncated, localities=localities)
    if all_ids or localities:
        country_cache.set(key, result)
    return result


# 3. Retrieve one mineral by ID  (declared last among /api/minerals/... so that "near" and "in-country" are not read as an id)
@app.get("/api/minerals/{mineral_id}")
def get_mineral(mineral_id: int):
    return mindat_request(f"geomaterials/{mineral_id}")


# 5. Retrieve localities and associated minerals (one page)
@app.get("/api/localities/with-minerals")
def localities_with_minerals_page(
    page_size: int = Query(100, ge=1, le=500),
    page: int = Query(1, ge=1),
    country: Optional[str] = Query(None, max_length=100),
):
    params = {"expand": "geomaterials", "fields": "id,txt,geomaterials", "page-size": page_size, "page": page}
    if country:
        params["country"] = country
    return mindat_request("localities", params)
