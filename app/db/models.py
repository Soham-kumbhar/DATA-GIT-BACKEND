from datetime import datetime

from sqlalchemy import DateTime, String, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
        unique=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class Version(Base):
    __tablename__ = "versions"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )

    # ML run associated with this DataGit version.
    # Nullable so existing versions remain valid.
    ml_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("ml_runs.id"),
        nullable=True,
        index=True,
    )

    version_number: Mapped[int] = mapped_column(
        nullable=False,
    )

    # Exact Git commit for this DataGit version
    git_commit: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    # Exact DVC state for this DataGit version
    dvc_state: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # Human-readable message describing why this version was finalized.
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class MLRun(Base):
    __tablename__ = "ml_runs"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )

    # Exact Git state when this ML run happened
    git_commit: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    # DVC state when this ML run happened
    dvc_state: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # ML information
    model_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    features: Mapped[list | None] = mapped_column(
        JSON,
        nullable=True,
    )

    parameters: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # Example: {"rmse": 42000, "r2": 0.94}
    metrics: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # Evaluation evidence used for detailed error analysis.
    # Example:
    # {
    #     "task_type": "classification",
    #     "confusion_matrix": {
    #         "tp": 3,
    #         "tn": 3,
    #         "fp": 0,
    #         "fn": 0
    #     },
    #     "precision": 1.0,
    #     "recall": 1.0,
    #     "f1": 1.0,
    #     "prediction_count": 6,
    #     "incorrect_count": 0,
    #     "error_rate": 0.0,
    #     "misclassified_samples": []
    # }
    evaluation: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )