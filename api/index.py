"""Vercel entrypoint for the MemFront/NDPA FastAPI app.

Vercel routes `/api/*` requests here. Some Vercel Python deployments also send
page requests to the ASGI entrypoint, so this wrapper serves the static landing
page/console directly before mounting the reusable API under `/api`.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.main import app as ndpa_app

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"

app = FastAPI(title="NDPA Vercel Entrypoint")
app.mount("/api", ndpa_app)
app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def landing_page():
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/console", include_in_schema=False)
@app.get("/console.html", include_in_schema=False)
def console_page():
    return FileResponse(WEB_ROOT / "console.html")


@app.get("/{path:path}", include_in_schema=False)
def single_page_fallback(path: str):
    if path.startswith("api/"):
        return {"detail": "Not Found"}
    return FileResponse(WEB_ROOT / "index.html")
