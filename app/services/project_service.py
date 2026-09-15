from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import Project
from app.schemas.project import ProjectCreate


class ProjectService:

    @staticmethod
    def create_project(
        db: Session,
        project_data: ProjectCreate,
    ) -> Project:

        project_path = Path(
            project_data.path
        ).expanduser()

        if not project_path.exists():
            raise ValueError(
                "Project path does not exist."
            )

        if not project_path.is_dir():
            raise ValueError(
                "Project path is not a directory."
            )

        resolved_path = str(
            project_path.resolve()
        )

        existing_project = (
            db.query(Project)
            .filter(Project.path == resolved_path)
            .first()
        )

        if existing_project:
            raise ValueError(
                "This project is already registered."
            )

        project = Project(
            name=project_data.name,
            path=resolved_path,
            description=project_data.description,
        )

        db.add(project)
        db.commit()
        db.refresh(project)

        return project

    @staticmethod
    def get_projects(
        db: Session,
    ) -> list[Project]:

        return db.query(Project).all()

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