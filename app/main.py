import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.projects import router as projects_router
from app.api.datasets import router as datasets_router
from app.api.models import router as models_router
from app.api.status import router as status_router

from app.api.dataset_preparation import (
    router as dataset_preparation_router,
)

from app.api.version_comparison import (
    router as version_comparison_router,
)

from app.api.version_history import (
    router as version_history_router,
)

from app.api.version_report import (
    router as version_report_router,
)

from app.api.versions import (
    router as versions_router,
)

from app.api.evidence import (
    router as evidence_router,
)

from app.api.runs import (
    router as runs_router,
)

from app.api.comparison import (
    router as comparison_router,
)

from app.core.config import settings
from app.db.database import Base, engine
from app.db import models


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)


# ============================================================
# CORS
# ============================================================
# Allowed origins now come from the ALLOWED_ORIGINS env var
# (comma-separated), falling back to localhost for local dev.
# Set ALLOWED_ORIGINS on your backend host to include your
# deployed frontend URL, e.g.:
#   ALLOWED_ORIGINS=https://frontend-zeta-one-69.vercel.app,http://localhost:5173

_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "application": settings.app_name,
    }


# ============================================================
# API ROUTERS
# ============================================================

app.include_router(
    projects_router
)

app.include_router(
    datasets_router
)

app.include_router(
    models_router
)

app.include_router(
    status_router
)

app.include_router(
    dataset_preparation_router
)


# ============================================================
# VERSION ROUTES
#
# The dedicated version comparison router already provides:
#
# GET
# /projects/{project_id}/versions/compare
#
# with:
#   version_1
#   version_2
#
# Keep this route in the version-scoped API rather than creating
# a second project-level comparison endpoint.
# ============================================================

app.include_router(
    version_comparison_router
)

app.include_router(
    version_history_router
)

app.include_router(
    version_report_router
)

app.include_router(
    versions_router
)


# ============================================================
# OTHER ROUTES
# ============================================================

app.include_router(
    evidence_router
)

app.include_router(
    runs_router
)

app.include_router(
    comparison_router
)
