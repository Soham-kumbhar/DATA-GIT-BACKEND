from sqlalchemy.orm import Session

from app.db.models import Project, Version
from app.schemas.version_history import VersionHistoryResponse


class VersionHistoryService:

    @staticmethod
    def get_history(
        db: Session,
        project_id: int,
    ) -> VersionHistoryResponse:

        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if project is None:
            raise ValueError("Project not found.")

        versions = (
            db.query(Version)
            .filter(
                Version.project_id == project_id
            )
            .order_by(
                Version.version_number.asc()
            )
            .all()
        )

        return VersionHistoryResponse(
            project_id=project_id,
            total_versions=len(versions),
            versions=versions,
        )