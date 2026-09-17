from dataclasses import dataclass
from datetime import datetime, timezone

from app.services.git_service import GitService
from app.services.dvc_service import DVCService


@dataclass(frozen=True)
class FinalizationEvidence:
    """Deterministic evidence captured when the user finalizes a version.

    This service deliberately does not create or track runs/experiments.
    """

    captured_at: str
    git: dict
    dvc: dict
    user_context: dict


class FinalizeVersionService:
    """Capture the current local project state for an intentional version."""

    @staticmethod
    def capture_evidence(
        project_path: str,
        message: str | None = None,
        notes: str | None = None,
    ) -> FinalizationEvidence:
        git_repository = GitService.is_git_repository(project_path)

        if git_repository:
            git = {
                "is_repository": True,
                "branch": GitService.get_current_branch(project_path),
                "commit": GitService.get_current_commit(project_path),
                "status": GitService.get_status(project_path),
            }
        else:
            git = {
                "is_repository": False,
                "branch": None,
                "commit": None,
                "status": None,
            }

        dvc_repository = DVCService.is_dvc_repository(project_path)
        if dvc_repository:
            dvc = {
                "is_repository": True,
                "status": DVCService.get_status(project_path),
                "diff": DVCService.get_diff(project_path),
                "tracked_files": DVCService.get_tracked_files(project_path),
            }
        else:
            dvc = {
                "is_repository": False,
                "status": None,
                "diff": None,
                "tracked_files": [],
            }

        return FinalizationEvidence(
            captured_at=datetime.now(timezone.utc).isoformat(),
            git=git,
            dvc=dvc,
            user_context={
                "message": message,
                "notes": notes,
            },
        )
