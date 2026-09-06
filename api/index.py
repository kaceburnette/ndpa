"""Vercel entrypoint for the MemFront/NDPA FastAPI app.

Vercel routes `/api/*` requests here. Some Vercel Python deployments also send
page requests to the ASGI entrypoint, so this wrapper serves the static landing
page/console directly before mounting the reusable API under `/api`.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.main import (
    DateNightEasyJoin,
    DateNightInviteCreate,
    DateNightMessageCreate,
    DateNightRoomCreate,
    app as ndpa_app,
    date_night_assistant,
    date_night_create_invite,
    date_night_create_room,
    date_night_easy_join,
    date_night_me,
    date_night_room,
    date_night_rooms,
    date_night_send_message,
    date_night_signout,
)

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"

app = FastAPI(title="NDPA Vercel Entrypoint")
app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def landing_page():
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/console", include_in_schema=False)
@app.get("/console.html", include_in_schema=False)
def console_page():
    return FileResponse(WEB_ROOT / "console.html")


@app.api_route("/api/index.py", methods=["GET", "POST"], include_in_schema=False)
async def vercel_date_night_proxy(request: Request):
    """Dispatch Date Night API calls after Vercel's single-function rewrite."""
    route = request.query_params.get("route", "")
    response = Response()
    try:
        if request.method == "GET" and route == "/date-night/me":
            result = await date_night_me(request)
        elif request.method == "GET" and route == "/date-night/rooms":
            result = await date_night_rooms(request)
        elif request.method == "GET" and route.startswith("/date-night/rooms/"):
            result = await date_night_room(route.rsplit("/", 1)[-1], request)
        elif request.method == "POST" and route == "/date-night/easy-join":
            result = await date_night_easy_join(DateNightEasyJoin(**await request.json()), request, response)
        elif request.method == "POST" and route == "/date-night/signout":
            result = await date_night_signout(request, response)
        elif request.method == "POST" and route == "/date-night/rooms":
            result = await date_night_create_room(DateNightRoomCreate(**await request.json()), request)
        elif request.method == "POST" and route.endswith("/messages"):
            result = await date_night_send_message(route.split("/")[-2], DateNightMessageCreate(**await request.json()), request)
        elif request.method == "POST" and route.endswith("/invites"):
            result = await date_night_create_invite(route.split("/")[-2], DateNightInviteCreate(**await request.json()), request)
        elif request.method == "POST" and route.endswith("/assistant"):
            result = await date_night_assistant(route.split("/")[-2], request)
        else:
            raise HTTPException(status_code=404, detail="Not Found")
    except HTTPException as exc:
        raise exc
    payload = JSONResponse(result)
    if "set-cookie" in response.headers:
        payload.headers.append("set-cookie", response.headers["set-cookie"])
    return payload


app.mount("/api", ndpa_app)
