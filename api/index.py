"""Vercel entrypoint for the NDPA FastAPI app.

Vercel routes `/api/*` requests here. The wrapper mounts the real FastAPI app
under `/api` so `/api/health` maps to the server's `/health` endpoint while
leaving the server package reusable for Fly/Docker/self-host deployments.
"""

from fastapi import FastAPI

from server.main import app as ndpa_app

app = FastAPI(title="NDPA Vercel Entrypoint")
app.mount("/api", ndpa_app)
app.mount("/", ndpa_app)
