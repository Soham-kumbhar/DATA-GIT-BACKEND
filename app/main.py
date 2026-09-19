import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.api.auth import router as auth_router
from app.api.projects import router as projects_router
from app.api.datasets import router as datasets_router
from app.api.models import router as models_router
from app.api.status import router as status_router

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
from app.db import user_models


# ============================================================
# OPTIONAL DATASET PREPARATION ROUTER
# ============================================================
#
# Dataset preparation currently imports scikit-learn.
# On this Windows machine, Application Control is blocking
# one of scikit-learn's native DLLs.
#
# We keep the rest of DATAGIT available so the API can start.
# The dataset-preparation router is included only when its
# dependencies can be imported successfully.
# ============================================================

dataset_preparation_router = None

try:
    from app.api.dataset_preparation import (
        router as dataset_preparation_router,
    )
except Exception as exc:
    print(
        "[WARNING] Dataset preparation router could not be loaded."
    )
    print(
        f"[WARNING] Reason: {exc}"
    )
    print(
        "[WARNING] The rest of DATAGIT will continue to start."
    )


# ============================================================
# SMALL SQLite MIGRATION
# ============================================================

def ensure_project_user_id_column():
    inspector = inspect(engine)

    tables = inspector.get_table_names()

    if "projects" not in tables:
        return

    columns = {
        column["name"]
        for column in inspector.get_columns("projects")
    }

    if "user_id" not in columns:
        if engine.dialect.name != "sqlite":
            raise RuntimeError(
                "The existing database needs a migration "
                "for projects.user_id."
            )

        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE projects "
                    "ADD COLUMN user_id INTEGER"
                )
            )


ensure_project_user_id_column()


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

default_origins = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "https://datagit-frontend.vercel.app"
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        default_origins,
    ).split(",")
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
        "dataset_preparation_available": (
            dataset_preparation_router is not None
        ),
    }


# ============================================================
# AUTHENTICATION
# ============================================================

app.include_router(
    auth_router
)


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

if dataset_preparation_router is not None:
    app.include_router(
        dataset_preparation_router
    )

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

app.include_router(
    evidence_router
)

app.include_router(
    runs_router
)

app.include_router(
    comparison_router
)