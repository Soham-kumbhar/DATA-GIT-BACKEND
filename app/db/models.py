from datetime import datetime

from sqlalchemy import (
    DateTime,
    JSON,
    String,
    Text,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.db.database import Base


class Project(Base):
    __tablename__ = "projects"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "project_number",
            name="uq_projects_user_project_number",
        ),
    )

    # ========================================================
    # INTERNAL DATABASE ID
    # ========================================================

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    # ========================================================
    # PROJECT OWNER
    #
    # Nullable only for legacy projects created before
    # authentication/project ownership existed.
    # ========================================================

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    # ========================================================
    # USER-FACING PROJECT NUMBER
    #
    # Starts from 1 independently for each user.
    #
    # Example:
    #
    # User A
    #   Project 1
    #   Project 2
    #
    # User B
    #   Project 1
    #   Project 2
    #
    # This is NOT the database primary key.
    # ========================================================

    project_number: Mapped[int | None] = mapped_column(
        nullable=True,
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

    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "version_number",
            name="uq_versions_project_version_number",
        ),
    )

    # ========================================================
    # INTERNAL DATABASE ID
    # ========================================================

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    # ========================================================
    # INTERNAL PROJECT FOREIGN KEY
    # ========================================================

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )

    # ========================================================
    # PROJECT RELATIONSHIP
    #
    # This allows VersionResponse to expose project_number
    # without changing project_id.
    # ========================================================

    project: Mapped["Project"] = relationship(
        "Project",
        lazy="joined",
    )

    @property
    def project_number(self) -> int | None:
        if self.project is None:
            return None

        return self.project.project_number

    # ========================================================
    # LEGACY ML RUN ASSOCIATION
    # ========================================================

    ml_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("ml_runs.id"),
        nullable=True,
        index=True,
    )

    # ========================================================
    # USER-FACING VERSION NUMBER
    #
    # Starts from 1 independently for each project.
    # ========================================================

    version_number: Mapped[int] = mapped_column(
        nullable=False,
    )

    # ========================================================
    # GIT STATE
    # ========================================================

    git_commit: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    # ========================================================
    # DVC STATE
    # ========================================================

    dvc_state: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # ========================================================
    # VERSION DESCRIPTION
    # ========================================================

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # ========================================================
    # PREPARATION EVIDENCE
    # ========================================================

    preparation_evidence = relationship(
        "VersionPreparationEvidence",
        back_populates="version",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # ========================================================
    # RESULT EVIDENCE
    # ========================================================

    result_evidence = relationship(
        "VersionResultEvidence",
        back_populates="version",
        uselist=False,
        cascade="all, delete-orphan",
    )


class VersionPreparationEvidence(Base):
    __tablename__ = "version_preparation_evidence"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("versions.id"),
        nullable=False,
        unique=True,
        index=True,
    )

    operations: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    version = relationship(
        "Version",
        back_populates="preparation_evidence",
    )


class VersionResultEvidence(Base):
    __tablename__ = "version_result_evidence"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("versions.id"),
        nullable=False,
        unique=True,
        index=True,
    )

    # ========================================================
    # MODEL EVIDENCE
    # ========================================================

    model_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    model_path: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True,
    )

    model_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    # ========================================================
    # FRAMEWORK EVIDENCE
    # ========================================================

    framework: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    framework_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # ========================================================
    # EVALUATION EVIDENCE
    # ========================================================

    evaluation: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    version = relationship(
        "Version",
        back_populates="result_evidence",
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

    git_commit: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    dvc_state: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

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

    metrics: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    evaluation: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )