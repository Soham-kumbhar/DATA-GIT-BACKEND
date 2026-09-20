import os
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
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


class ProjectConflictError(ValueError):

    def __init__(
        self,
        message: str,
        code: str,
        project_id: int | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.code = code
        self.project_id = project_id


class ProjectService:

    @staticmethod
    def normalize_project_path(
        path: str,
    ) -> str:

        value = path.strip()

        if not value:
            return value

        if value.lower().startswith(
            "datagit://"
        ):
            return value

        return os.path.normcase(
            os.path.abspath(
                os.path.normpath(value)
            )
        )

    @staticmethod
    def _find_project_by_path(
        db: Session,
        project_path: str,
    ) -> Project | None:

        return (
            db.query(Project)
            .filter(
                Project.path == project_path
            )
            .first()
        )

    @staticmethod
    def _get_next_project_number(
        db: Session,
        user_id: int,
    ) -> int:

        current_max = (
            db.query(
                func.max(
                    Project.project_number
                )
            )
            .filter(
                Project.user_id == user_id
            )
            .scalar()
        )

        if current_max is None:
            return 1

        return current_max + 1

    @staticmethod
    def create_project(
        db: Session,
        user_id: int,
        project_data: ProjectCreate,
    ) -> Project:

        project_name = (
            project_data.name.strip()
        )

        if not project_name:
            raise ValueError(
                "Project name cannot be empty."
            )

        if project_data.path:
            project_path = (
                ProjectService.normalize_project_path(
                    project_data.path
                )
            )
        else:
            project_path = (
                f"datagit://project/"
                f"{uuid4().hex}"
            )

        if not project_path:
            project_path = (
                f"datagit://project/"
                f"{uuid4().hex}"
            )

        existing_name = (
            db.query(Project)
            .filter(
                Project.user_id == user_id,
                Project.name == project_name,
            )
            .first()
        )

        if existing_name:
            raise ProjectConflictError(
                message=(
                    "You already have a project "
                    "with this name."
                ),
                code=(
                    "PROJECT_NAME_ALREADY_EXISTS"
                ),
                project_id=existing_name.id,
            )

        existing_path = (
            ProjectService._find_project_by_path(
                db,
                project_path,
            )
        )

        if existing_path:

            if (
                existing_path.user_id
                == user_id
            ):
                raise ProjectConflictError(
                    message=(
                        "This folder is already "
                        "registered to your account."
                    ),
                    code=(
                        "PROJECT_PATH_ALREADY_REGISTERED"
                    ),
                    project_id=existing_path.id,
                )

            if existing_path.user_id is None:
                raise ProjectConflictError(
                    message=(
                        "This folder belongs to a "
                        "legacy DATAGIT project created "
                        "before account ownership was enabled."
                    ),
                    code=(
                        "LEGACY_PROJECT_UNCLAIMED"
                    ),
                    project_id=existing_path.id,
                )

            raise ProjectConflictError(
                message=(
                    "This folder is already registered "
                    "to another DATAGIT account."
                ),
                code=(
                    "PROJECT_PATH_OWNED_BY_OTHER_USER"
                ),
            )

        project_number = (
            ProjectService._get_next_project_number(
                db,
                user_id,
            )
        )

        project = Project(
            user_id=user_id,
            project_number=project_number,
            name=project_name,
            path=project_path,
            description=(
                project_data.description.strip()
                if project_data.description
                else None
            ),
        )

        db.add(project)

        try:
            db.commit()
            db.refresh(project)

        except IntegrityError as error:
            db.rollback()

            existing_path = (
                ProjectService._find_project_by_path(
                    db,
                    project_path,
                )
            )

            if existing_path:

                if (
                    existing_path.user_id
                    == user_id
                ):
                    raise ProjectConflictError(
                        message=(
                            "This folder is already "
                            "registered to your account."
                        ),
                        code=(
                            "PROJECT_PATH_ALREADY_REGISTERED"
                        ),
                        project_id=existing_path.id,
                    ) from error

                if existing_path.user_id is None:
                    raise ProjectConflictError(
                        message=(
                            "This folder belongs to a "
                            "legacy DATAGIT project created "
                            "before account ownership was enabled."
                        ),
                        code=(
                            "LEGACY_PROJECT_UNCLAIMED"
                        ),
                        project_id=existing_path.id,
                    ) from error

                raise ProjectConflictError(
                    message=(
                        "This folder is already registered "
                        "to another DATAGIT account."
                    ),
                    code=(
                        "PROJECT_PATH_OWNED_BY_OTHER_USER"
                    ),
                ) from error

            raise

        return project

    @staticmethod
    def claim_legacy_project(
        db: Session,
        user_id: int,
        project_path: str,
    ) -> Project:

        normalized_path = (
            ProjectService.normalize_project_path(
                project_path
            )
        )

        existing_project = (
            ProjectService._find_project_by_path(
                db,
                normalized_path,
            )
        )

        if existing_project is None:
            raise ProjectConflictError(
                message=(
                    "No legacy project was found "
                    "for this folder."
                ),
                code=(
                    "LEGACY_PROJECT_NOT_FOUND"
                ),
            )

        if existing_project.user_id == user_id:
            return existing_project

        if existing_project.user_id is not None:
            raise ProjectConflictError(
                message=(
                    "This folder is already registered "
                    "to another DATAGIT account."
                ),
                code=(
                    "PROJECT_PATH_OWNED_BY_OTHER_USER"
                ),
            )

        existing_project.user_id = user_id

        existing_project.project_number = (
            ProjectService._get_next_project_number(
                db,
                user_id,
            )
        )

        try:
            db.commit()
            db.refresh(existing_project)

        except IntegrityError as error:
            db.rollback()

            raise ProjectConflictError(
                message=(
                    "The legacy project could not be "
                    "claimed because its ownership changed."
                ),
                code=(
                    "LEGACY_PROJECT_CLAIM_CONFLICT"
                ),
            ) from error

        return existing_project

    @staticmethod
    def get_projects(
        db: Session,
        user_id: int,
    ) -> list[Project]:

        return (
            db.query(Project)
            .filter(
                Project.user_id == user_id
            )
            .order_by(
                Project.project_number.asc()
            )
            .all()
        )

    @staticmethod
    def get_project(
        db: Session,
        user_id: int,
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
        user_id: int,
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

            db.query(Version).filter(
                Version.project_id
                == project_id
            ).delete(
                synchronize_session=False
            )

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

            db.query(MLRun).filter(
                MLRun.project_id
                == project_id
            ).delete(
                synchronize_session=False
            )

            db.delete(project)
            db.commit()

            return True

        except Exception:
            db.rollback()
            raise