import os
import tempfile
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.db.models import Project
from app.schemas.project import ProjectCreate


class ProjectService:

    @staticmethod
    def create_project(
        db: Session,
        project_data: ProjectCreate,
    ) -> Project:

        project_name = project_data.name.strip()

        if not project_name:
            raise ValueError("Project name cannot be empty.")

        configured_root = os.getenv("DATAGIT_PROJECT_ROOT")

        if configured_root:
            project_root = Path(configured_root).expanduser()
        else:
            project_root = Path(tempfile.gettempdir()) / "datagit-projects"

        project_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        project_path = project_root / uuid4().hex

        project_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        resolved_path = str(project_path.resolve())

        existing_project = (
            db.query(Project)
            .filter(Project.path == resolved_path)
            .first()
        )

        if existing_project:
            raise ValueError(
                "Unable to create a unique project workspace. Please try again."
            )

        project = Project(
            name=project_name,
            path=resolved_path,
            description=project_data.description,
        )

        db.add(project)

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        db.refresh(project)

        return project

    @staticmethod
    def get_projects(
        db: Session,
    ) -> list[Project]:

        return (
            db.query(Project)
            .order_by(Project.created_at.desc())
            .all()
        )

    @staticmethod
    def get_project(
        db: Session,
        project_id: int,
    ) -> Project | None:

        return (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )
