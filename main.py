
import os
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mindat GIS API")

MINDAT_API_URL = "https://api.mindat.org"
MINDAT_API_KEY = os.getenv("MINDAT_API_KEY")

# Add your GIS application's domain in Render environment
# Example: https://my-gis-app.onrender.com
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "*"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def mindat_request(endpoint, params=None):
    if not MINDAT_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Mindat API key is not configured"
        )

    headers = {
        "Authorization": f"Token {MINDAT_API_KEY}"
    }

    params = params or {}
    params["format"] = "json"

    try:
        response = requests.get(
            f"{MINDAT_API_URL}/{endpoint}/",
            params=params,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Mindat API request timed out"
        )

    except requests.RequestException as e:
        status = (
            e.response.status_code
            if e.response is not None else 502
        )
        raise HTTPException(
            status_code=502,
            detail=f"Mindat API request failed (upstream status: {status})"
        )

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Mindat returned invalid JSON"
        )


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Mindat GIS API"
    }


# 1. Search minerals by name
@app.get("/api/minerals/search")
def search_minerals(
    q: str = Query(..., min_length=2, max_length=100),
    size: int = Query(12, ge=1, le=50)
):
    return mindat_request(
        "items_search",
        {
            "q": q,
            "size": size
        }
    )


# 2. Retrieve mineral information
@app.get("/api/minerals")
def get_minerals(
    name: str = None,
    ima: bool = None,
    page_size: int = Query(100, ge=1, le=500)
):
    params = {
        "fields": "id,name,dispformulasimple",
        "page_size": page_size
    }

    if name:
        params["name__icontains"] = name

    if ima is not None:
        params["ima"] = int(ima)

    return mindat_request("items", params)


# 3. Retrieve one mineral by ID
@app.get("/api/minerals/{mineral_id}")
def get_mineral(mineral_id: int):
    return mindat_request(
        f"items/{mineral_id}",
        {"expand": "relations"}
    )


# 4. Search localities
@app.get("/api/localities/search")
def search_localities(
    txt: str = Query(None, max_length=150),
    country: str = Query(None, max_length=100),
    page_size: int = Query(100, ge=1, le=500)
):
    params = {
        "page_size": page_size,
        "fields": "id,name",
    }

    if txt:
        params["txt"] = txt

    if country:
        params["country"] = country

    return mindat_request("localities", params)


# 5. Retrieve localities and associated minerals
@app.get("/api/localities/with-minerals")
def localities_with_minerals(
    page_size: int = Query(100, ge=1, le=500)
):
    return mindat_request(
        "localities",
        {
            "expand": "items",
            "fields": "id,name,items",
            "page_size": page_size
        }
    )
