"""FastAPI entry point.  Run from the backend folder:  uvicorn app.main:app --reload"""
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from .api.routes import router
from .core.config import get_settings
from .core.errors import AppError
from .database.session import init_db

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("maize")
s = get_settings()
app = FastAPI(title="UAV-Based Intelligent Maize Crop Health Monitoring")
app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])
init_db()
(s.results_dir / "uav").mkdir(parents=True, exist_ok=True)
app.mount("/files/uav", StaticFiles(directory=s.results_dir / "uav"), name="uav_overlays")   # random filenames only
app.include_router(router)


@app.exception_handler(AppError)
async def app_error(_: Request, e: AppError):
    return JSONResponse(status_code=e.status, content={"error": {"code": e.code, "message": e.message}})


@app.exception_handler(Exception)
async def unexpected(_: Request, e: Exception):
    log.exception("Unhandled error")     # full trace goes to the server log only
    return JSONResponse(status_code=500, content={"error": {"code": "server_error",
                        "message": "Something went wrong on the server. Please try again."}})
