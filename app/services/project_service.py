from uuid import uuid4

from sqlalchemy.orm import Session

from app.db.models import (
    Dataset,
    MLRun,
    Model,
    Project,
    Version,
    VersionPreparationEvidence,
    VersionResultEvidence,
)
from app.schemas.project import ProjectCreate


class ProjectService:

    @staticmethod
    def create_project(
        db: Session,
        user_id: int | None,
        project_data: ProjectCreate,
    ) -> Project:

        project_name = project_data.name.strip()

        if not project_name:
            raise ValueError(
                "Project name cannot be empty."
            )

        if project_data.path:
            project_path = project_data.path.strip()
        else:
            project_path = (
                f"datagit://project/{uuid4().hex}"
            )

        if not project_path:
            project_path = (
                f"datagit://project/{uuid4().hex}"
            )

        existing_project = (
            db.query(Project)
            .filter(
                Project.name == project_name,
            )
            .first()
        )

        if existing_project:
            raise ValueError(
                "You already have a project with this name."
            )

        project = Project(
            user_id=user_id,
            name=project_name,
            path=project_path,
            description=project_data.description,
        )

        db.add(project)

        try:
            db.commit()
            db.refresh(project)
        except Exception:
            db.rollback()
            raise

        return project

    @staticmethod
    def get_projects(
        db: Session,
        user_id: int | None,
    ) -> list[Project]:

        return (
            db.query(Project)
            .order_by(Project.created_at.desc())
            .all()
        )

    @staticmethod
    def get_project(
        db: Session,
        user_id: int | None,
        project_id: int,
    ) -> Project | None:

        return (
            db.query(Project)
            .filter(
                Project.id == project_id,
                Project.user_id == user_id,
            )
            .first()
        )

    @staticmethod
    def delete_project(
        db: Session,
        user_id: int | None,
        project_id: int,
    ) -> bool:

        project = (
            db.query(Project)
            .filter(
                Project.id == project_id,
                Project.user_id == user_id,
            )
            .first()
        )

        if project is None:
            return False

        try:
            # ------------------------------------------------
            # Delete version evidence first.
            # ------------------------------------------------

            version_ids = [
                row[0]
                for row in (
                    db.query(Version.id)
                    .filter(
                        Version.project_id
                        == project_id
                    )
                    .all()
                )
            ]

            if version_ids:
                db.query(
                    VersionResultEvidence
                ).filter(
                    VersionResultEvidence.version_id.in_(
                        version_ids
                    )
                ).delete(
                    synchronize_session=False
                )

                db.query(
                    VersionPreparationEvidence
                ).filter(
                    VersionPreparationEvidence.version_id.in_(
                        version_ids
                    )
                ).delete(
                    synchronize_session=False
                )

            # ------------------------------------------------
            # Delete finalized versions.
            # ------------------------------------------------

            db.query(Version).filter(
                Version.project_id
                == project_id
            ).delete(
                synchronize_session=False
            )

            # ------------------------------------------------
            # Delete project-owned datasets/models.
            # ------------------------------------------------

            db.query(Dataset).filter(
                Dataset.project_id
                == project_id
            ).delete(
                synchronize_session=False
            )

            db.query(Model).filter(
                Model.project_id
                == project_id
            ).delete(
                synchronize_session=False
            )

            # ------------------------------------------------
            # Delete legacy ML runs for this project.
            # ------------------------------------------------

            db.query(MLRun).filter(
                MLRun.project_id
                == project_id
            ).delete(
                synchronize_session=False
            )

            # ------------------------------------------------
            # Finally delete the project itself.
            # ------------------------------------------------

            db.delete(project)
            db.commit()

            return True

        except Exception:
            db.rollback()
            raise