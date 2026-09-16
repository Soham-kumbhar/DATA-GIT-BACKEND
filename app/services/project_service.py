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

        if project_path.exists() and not project_path.is_dir():
            raise ValueError(
                "Project path exists but is not a directory."
            )

        # NOTE: this creates the folder on the *backend server's*
        # filesystem, not on the machine the browser is running on.
        # On an ephemeral host (Render free tier, etc.) this folder
        # will not persist across restarts/redeploys, and it will
        # not contain any real Git/DVC repo. This unblocks the
        # "Create Project" flow for a remotely-deployed demo; it
        # does not give DATAGIT real access to a user's local repo.
        project_path.mkdir(parents=True, exist_ok=True)

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
