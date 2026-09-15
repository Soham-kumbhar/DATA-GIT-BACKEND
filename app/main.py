from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.compare import router as compare_router
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
# Allow the React/Vite development server to communicate
# with this FastAPI backend from a browser.
#
# Vite normally runs on:
#   http://localhost:5173
#   http://127.0.0.1:5173
#
# Keep origins explicit rather than using "*".

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
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
# Specific routes such as /compare, /history and /report
# are registered before /versions/{version_id}.
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
app.include_router(compare_router)