import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.api.auth import router as auth_router
from app.api.projects import router as projects_router
from app.api.datasets import router as datasets_router
from app.api.models import router as models_router
from app.api.status import router as status_router
from app.api.system import router as system_router

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

dataset_preparation_router = None

try:
    from app.api.dataset_preparation import (
        router as dataset_preparation_router,
    )
except Exception as exc:
    print(
        "[WARNING] Dataset preparation router "
        "could not be loaded."
    )
    print(
        f"[WARNING] Reason: {exc}"
    )
    print(
        "[WARNING] The rest of DATAGIT will continue to start."
    )


# ============================================================
# SQLITE MIGRATION
# ============================================================

def ensure_project_user_id_column():
    inspector = inspect(engine)

    tables = inspector.get_table_names()

    if "projects" not in tables:
        return

    columns = {
        column["name"]
        for column in inspector.get_columns(
            "projects"
        )
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


def ensure_user_oauth_columns():
    inspector = inspect(engine)

    tables = inspector.get_table_names()

    if "users" not in tables:
        return

    columns = {
        column["name"]
        for column in inspector.get_columns(
            "users"
        )
    }

    # The current DATAGIT database is SQLite.
    # Keep the migration explicit for other databases.
    if engine.dialect.name != "sqlite":
        missing_columns = (
            {
                "google_id",
                "github_id",
            }
            - columns
        )

        if missing_columns:
            raise RuntimeError(
                "The existing database needs a migration "
                "for users OAuth columns: "
                f"{sorted(missing_columns)}"
            )

        return

    with engine.begin() as connection:

        if "google_id" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE users "
                    "ADD COLUMN google_id VARCHAR(255)"
                )
            )

        if "github_id" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE users "
                    "ADD COLUMN github_id VARCHAR(255)"
                )
            )

        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "ix_users_google_id "
                "ON users (google_id)"
            )
        )

        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "ix_users_github_id "
                "ON users (github_id)"
            )
        )


ensure_project_user_id_column()

ensure_user_oauth_columns()


# ============================================================
# DATABASE TABLES
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

fastapi_app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)


# ============================================================
# ROUTERS
# ============================================================

fastapi_app.include_router(
    auth_router
)

fastapi_app.include_router(
    projects_router
)

fastapi_app.include_router(
    datasets_router
)

fastapi_app.include_router(
    models_router
)

fastapi_app.include_router(
    status_router
)

if dataset_preparation_router is not None:
    fastapi_app.include_router(
        dataset_preparation_router
    )

fastapi_app.include_router(
    version_comparison_router
)

fastapi_app.include_router(
    version_history_router
)

fastapi_app.include_router(
    version_report_router
)

fastapi_app.include_router(
    versions_router
)

fastapi_app.include_router(
    evidence_router
)

fastapi_app.include_router(
    runs_router
)

fastapi_app.include_router(
    comparison_router
)

fastapi_app.include_router(
    system_router
)


# ============================================================
# HEALTH CHECK
# ============================================================

@fastapi_app.get("/health")
def health_check():
    return {
        "status": "ok",
        "application": settings.app_name,
        "dataset_preparation_available": (
            dataset_preparation_router is not None
        ),
    }


# ============================================================
# CORS
#
# Wrap the complete application.
# This guarantees CORS headers are still attached to
# unexpected 500 responses so the browser exposes the
# actual backend error instead of only saying:
# "Network Error".
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


app = CORSMiddleware(
    app=fastapi_app,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
