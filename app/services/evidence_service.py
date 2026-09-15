from app.services.git_service import GitService
from app.services.dvc_service import DVCService


class EvidenceService:

    @staticmethod
    def get_project_evidence(
        project_path: str,
        commit_hash: str | None = None,
    ) -> dict:

        evidence = {
            "project_path": project_path,
            "git": {},
            "dvc": {},
        }

        # -------------------------
        # Git evidence
        # -------------------------
        is_git = GitService.is_git_repository(project_path)

        evidence["git"]["is_repository"] = is_git

        if is_git:
            evidence["git"]["current_branch"] = (
                GitService.get_current_branch(project_path)
            )

            evidence["git"]["current_commit"] = (
                GitService.get_current_commit(project_path)
            )

            evidence["git"]["commits"] = (
                GitService.get_commits(project_path)
            )

            evidence["git"]["status"] = (
                GitService.get_status(project_path)
            )

            evidence["git"]["diff"] = (
                GitService.get_diff(project_path)
            )

            if commit_hash:
                evidence["git"]["commit_diff"] = (
                    GitService.get_commit_diff(
                        project_path,
                        commit_hash,
                    )
                )

        # -------------------------
        # DVC evidence
        # -------------------------
        is_dvc = DVCService.is_dvc_repository(project_path)

        evidence["dvc"]["is_repository"] = is_dvc

        if is_dvc:
            evidence["dvc"]["status"] = (
                DVCService.get_status(project_path)
            )

            evidence["dvc"]["diff"] = (
                DVCService.get_diff(project_path)
            )

        return evidence