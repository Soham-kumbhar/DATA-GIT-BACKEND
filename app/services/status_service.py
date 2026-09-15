from sqlalchemy.orm import Session

from app.db.models import Project, Version
from app.schemas.status import ProjectStatusResponse
from app.services.git_service import GitService
from app.services.dvc_service import DVCService


class StatusService:

    @staticmethod
    def get_project_status(
        db: Session,
        project_id: int,
    ) -> ProjectStatusResponse:

        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if project is None:
            raise ValueError("Project not found.")

        latest_version = (
            db.query(Version)
            .filter(Version.project_id == project_id)
            .order_by(Version.version_number.desc())
            .first()
        )

        # There is no baseline yet.
        if latest_version is None:
            return ProjectStatusResponse(
                project_id=project_id,
                latest_version=None,
                git_changed=False,
                dvc_changed=False,
                changes=["No DataGit version exists yet."],
            )

        project_path = project.path

        current_git_commit = GitService.get_current_commit(
            project_path
        )

        current_dvc_state = DVCService.get_state(
            project_path
        )

        git_changed = (
            current_git_commit
            != latest_version.git_commit
        )

        dvc_changed = (
            current_dvc_state
            != latest_version.dvc_state
        )

        changes = []

        if git_changed:
            changes.append("model/code changed")

        if dvc_changed:
            changes.append("dataset changed")

        if not changes:
            changes.append("No changes detected.")

        return ProjectStatusResponse(
            project_id=project_id,
            latest_version=latest_version.version_number,
            git_changed=git_changed,
            dvc_changed=dvc_changed,
            changes=changes,
        )