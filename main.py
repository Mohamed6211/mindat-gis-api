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


LOC_SAMPLE = {}                                             # mineral id -> short extract of the raw loc-by-min answer (diagnostic)


def _loc_id(x):
    """A locality id from a number, a numeric string or an object like {"id": 123}."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str) and x.strip().isdigit():
        return int(x)
    if isinstance(x, dict):
        for k in ("id", "loc_id", "locid", "locality_id", "locality", "loc"):
            if k in x:
                return _loc_id(x[k])
    return None


def _loc_items(data):
    """The list of items of a loc-by-min answer, whatever the envelope."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("results", "localities", "locs", "ids", "data", "loc", "locations"):
            if isinstance(data.get(k), list):
                return data[k]
        for v in data.values():                             # {"<mineral id>": [locality ids]}
            if isinstance(v, list):
                return v
    return []


def _localities_of_mineral(mineral_id):
    """Locality ids where ONE mineral has been reported (loc-by-min) -> (ids, error message or None)."""
    last_error = None
    for exc in ("0", None):                                 # the endpoint wants a non-empty "exc"
        params = {"inc": str(mineral_id)}
        if exc:
            params["exc"] = exc
        try:
            data = mindat_request("loc-by-min", params, timeout=60)
        except HTTPException as e:
            last_error = f"{e.status_code}: {e.detail}"
            continue
        LOC_SAMPLE[mineral_id] = {"type": type(data).__name__, "exc": exc, "extract": json.dumps(data)[:300]}
        ids = [i for i in (_loc_id(x) for x in _loc_items(data)) if i]
        if ids or exc is None:
            return ids, None
        # empty with exc=0: try again without "exc" before concluding there is nothing
    return [], last_error


def localities_with_minerals(ids):
    """Union of the localities of each mineral (one loc-by-min request per mineral: a list in "inc" means
    "localities that have ALL these minerals"). Returns (sorted ids, list of errors)."""
    found, errors = set(), []
    ids = list(ids)[:5]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for part, err in pool.map(_localities_of_mineral, ids):
            found.update(part)
            if err:
                errors.append(err)
    return sorted(found), errors


@app.get("/api/debug/loc-by-min")
def debug_loc_by_min(inc: int = Query(5473)):
    """Raw Mindat answers for loc-by-min (diagnostic: remove when everything works)."""
    if not MINDAT_API_KEY:
        raise HTTPException(status_code=500, detail="Mindat API key is not configured")
    headers = {"Authorization": f"Token {MINDAT_API_KEY}", "Accept": "application/json", "User-Agent": USER_AGENT}
    out = {}
    for label, params in (("exc=0", {"inc": inc, "exc": "0"}), ("no exc", {"inc": inc}), ("exc=1", {"inc": inc, "exc": "1"})):
        try:
            r = session.get(f"{MINDAT_API_URL}/loc-by-min/", params=params, headers=headers, timeout=60)
            out[label] = {"status": r.status_code, "content_type": r.headers.get("Content-Type"),
                          "length": len(r.text), "start": r.text[:300]}
        except requests.RequestException as e:
            out[label] = {"error": e.__class__.__name__}
    return out


def _probe(endpoint, params=None, method="GET", body=None):
    """One raw Mindat call summarised (diagnostic)."""
    headers = {"Authorization": f"Token {MINDAT_API_KEY}", "Accept": "application/json", "User-Agent": USER_AGENT}
    url = f"{MINDAT_API_URL}/{endpoint.strip('/')}/"
    try:
        r = session.request(method, url, params=params, json=body, headers=headers, timeout=60)
    except requests.RequestException as e:
        return {"error": e.__class__.__name__}
    out = {"status": r.status_code, "length": len(r.text)}
    try:
        data = r.json()
        if isinstance(data, dict):
            out["keys"] = list(data.keys())[:12]
            if "count" in data:
                out["count"] = data["count"]
            res = data.get("results")
            if isinstance(res, list):
                out["n_results"] = len(res)
        elif isinstance(data, list):
            out["n_items"] = len(data)
        out["start"] = json.dumps(data)[:350]
    except ValueError:
        out["start"] = r.text[:150]
    return out


@app.get("/api/debug/probe")
def debug_probe(mineral: str = "stibnite", country: str = "Spain"):
    """Battery of raw Mindat calls to see which way of 'mineral -> localities of a country' works (diagnostic)."""
    if not MINDAT_API_KEY:
        raise HTTPException(status_code=500, detail="Mindat API key is not configured")
    resolved = resolve_mineral_ids(mineral)
    mid = resolved[0][0] if resolved else None
    out = {"mineral": mineral, "resolved": resolved[:3]}
    if mid is None:
        return out
    out["1_loc_by_min_exc0"] = _probe("loc-by-min", {"inc": mid, "exc": "0"})
    out["2_loc_by_min_exc_other"] = _probe("loc-by-min", {"inc": mid, "exc": "1"})
    out["3_occurrences_by_min"] = _probe("occurrences", {"min": mid, "page-size": 3})
    out["4_localities_country"] = _probe("localities", {"country": country, "page-size": 2, "fields": "id,txt,latitude,longitude,country"})
    out["5_localities_country_expand"] = _probe("localities", {"country": country, "page-size": 2, "expand": "geomaterials", "fields": "id,txt,geomaterials"})
    out["6_localities_country_elements"] = _probe("localities", {"country": country, "elements_inc": "Au", "page-size": 1, "fields": "id,txt"})
    out["7_geomin_point_madrid"] = _probe("geomin-point", method="POST", body={"point": {"lat": 40.4, "lon": -3.7}, "distance": "50km"})
    return out


# ---------------------------------------------------------------------------
# Country scan: "localities of a country + the minerals of each locality" (localities?country=...&expand=geomaterials),
# read page by page ONCE per country and kept in memory; then any mineral of that country is answered instantly.
# (loc-by-min answers [] for very common minerals such as gold and only half of the rare ones can be checked.)
# ---------------------------------------------------------------------------
SCAN_PAGE_SIZE, SCAN_MAX_PAGES, SCAN_BATCH, SCAN_TTL = 300, 200, 6, 6 * 3600
SCANS = {}                                                 # country -> {"rows", "next_page", "complete", "errors", "updated"}
_scan_lock = threading.Lock()


def _gm_ids(items):
    out = set()
    for x in items or []:
        i = _loc_id(x)
        if i:
            out.add(i)
    return frozenset(out)


def _scan_page(country, page):
    """-> (rows, is_last_page); rows is None when the page could not be read."""
    params = {"country": country, "expand": "geomaterials", "page-size": SCAN_PAGE_SIZE, "page": page,
              "fields": "id,txt,latitude,longitude,country,geomaterials"}
    try:
        data = mindat_request("localities", params, timeout=45)
    except HTTPException as e:
        return ([], True) if e.status_code == 404 else (None, False)     # 404 = after the last page
    items = _results(data)
    rows = [(loc.get("id"), loc.get("txt"), loc.get("latitude"), loc.get("longitude"),
             loc.get("country") or country, _gm_ids(loc.get("geomaterials")))
            for loc in items if isinstance(loc, dict)]
    has_next = isinstance(data, dict) and bool(data.get("next"))
    return rows, (not items) or (isinstance(data, dict) and not has_next)


def advance_scan(country, budget=70.0):
    """Reads pages of the country until it is complete, the page cap is reached or the time budget is spent.
    Progress is kept: a later request goes on where this one stopped."""
    key = country.strip().lower()
    started = time.monotonic()
    with _scan_lock:
        st = SCANS.get(key)
        if st and st["complete"] and time.time() - st["updated"] > SCAN_TTL:
            st = None
        if not st:
            if len(SCANS) >= 8:                              # keep the memory small: drop the oldest country
                del SCANS[min(SCANS, key=lambda k: SCANS[k]["updated"])]
            st = SCANS[key] = {"rows": [], "next_page": 1, "complete": False, "errors": 0, "updated": time.time()}
        while not st["complete"] and st["next_page"] <= SCAN_MAX_PAGES and time.monotonic() - started < budget:
            pages = list(range(st["next_page"], min(st["next_page"] + SCAN_BATCH, SCAN_MAX_PAGES + 1)))
            with ThreadPoolExecutor(max_workers=SCAN_BATCH) as pool:
                results = list(pool.map(lambda p: _scan_page(country, p), pages))
            advanced = 0
            for rows, last in results:
                if rows is None:                             # unreadable page: stop here, it will be asked again
                    break
                st["rows"].extend(rows)
                advanced += 1
                if last:
                    st["complete"] = True
                    break
            st["next_page"] += advanced
            if advanced == 0:
                st["errors"] += 1
                if st["errors"] >= 3:
                    break
                time.sleep(1.0)
        st["updated"] = time.time()
        return st


def _via_loc_by_min(ids, country, started, max_chunks=40, chunk=100):
    """Old way (rare minerals only), used to complete a scan that could not be finished. -> (localities, errors)."""
    all_ids, errors = localities_with_minerals(ids)
    chunks = [all_ids[k:k + chunk] for k in range(0, len(all_ids), chunk)][:max_chunks]

    def fetch(ids_chunk):
        if time.monotonic() - started > 85:
            return []
        params = {"id_in": ",".join(str(i) for i in ids_chunk), "country": country,
                  "fields": "id,txt,latitude,longitude,country", "page-size": len(ids_chunk)}
        try:
            data = mindat_request("localities", params)
        except HTTPException:
            return []
        return [{"id": loc.get("id"), "txt": loc.get("txt"), "latitude": loc.get("latitude"),
                 "longitude": loc.get("longitude"), "country": loc.get("country") or country}
                for loc in _results(data) if isinstance(loc, dict)]

    out = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for part in pool.map(fetch, chunks):
            out.extend(part)
    return out, errors


@app.get("/api/minerals/in-country")
def minerals_in_country(
    mineral: str = Query(..., min_length=2, max_length=100),
    country: str = Query(..., min_length=2, max_length=100),
    max_localities: int = Query(1500, ge=1, le=5000),
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
    target = set(base["mineral_ids"])
    st = advance_scan(country.strip())
    complete = st["complete"]
    found = {}
    for lid, txt, lat, lon, ctry, gms in st["rows"]:
        if lid is not None and gms & target:
            found[lid] = {"id": lid, "txt": txt, "latitude": lat, "longitude": lon, "country": ctry}
    base.update(method="country-scan", scanned_localities=len(st["rows"]), scan_complete=complete)
    if not complete:                                         # country too big / Mindat slow: complete with loc-by-min
        extra, errors = _via_loc_by_min(base["mineral_ids"], country, started)
        base["loc_by_min_errors"] = errors
        for loc in extra:
            found.setdefault(loc["id"], loc)
    localities = list(found.values())
    truncated = (not complete) or len(localities) > max_localities
    total = len(localities)
    localities = localities[:max_localities]
    result = dict(base, count=len(localities), total_with_mineral=total, truncated=truncated, localities=localities)
    if complete:                                             # a partial answer is not kept: the next request goes on with the scan
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
