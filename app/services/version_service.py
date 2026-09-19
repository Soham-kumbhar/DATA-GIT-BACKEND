from sqlalchemy.orm import Session

from app.db.models import (
    Project,
    Version,
    VersionPreparationEvidence,
    VersionResultEvidence,
)
from app.services.dvc_service import DVCService
from app.services.git_service import GitService


class VersionService:

    @staticmethod
    def finalize_version(
        db: Session,
        project_id: int,
        description: str,
        preparation_operations: list[dict] | None = None,
        result_evidence: dict | None = None,
    ) -> Version:
        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if not project:
            raise ValueError("Project not found.")

        git_commit = GitService.get_current_commit(
            project.path
        )

        dvc_state = DVCService.get_state(
            project.path
        )

        existing_versions = (
            db.query(Version)
            .filter(
                Version.project_id == project_id
            )
            .order_by(
                Version.version_number.asc()
            )
            .all()
        )

        for existing_version in existing_versions:
            if (
                existing_version.git_commit
                == git_commit
                and existing_version.dvc_state
                == dvc_state
            ):
                raise ValueError(
                    "This Git + DVC state is already finalized as "
                    f"DataGit Version {existing_version.version_number}."
                )

        version_number = (
            1
            if not existing_versions
            else existing_versions[-1].version_number + 1
        )

        version = Version(
            project_id=project_id,
            version_number=version_number,
            git_commit=git_commit,
            dvc_state=dvc_state,
            description=description,
            ml_run_id=None,
        )

        db.add(version)
        db.flush()

        preparation_evidence = (
            VersionPreparationEvidence(
                version_id=version.id,
                operations=(
                    preparation_operations
                    if preparation_operations is not None
                    else []
                ),
            )
        )

        db.add(preparation_evidence)

        if result_evidence:
            db.add(
                VersionResultEvidence(
                    version_id=version.id,
                    model_name=result_evidence.get(
                        "model_name"
                    ),
                    model_path=result_evidence.get(
                        "model_path"
                    ),
                    model_sha256=result_evidence.get(
                        "model_sha256"
                    ),
                    framework=result_evidence.get(
                        "framework"
                    ),
                    framework_version=result_evidence.get(
                        "framework_version"
                    ),
                    metrics=result_evidence.get(
                        "metrics"
                    ),
                    evaluation=result_evidence.get(
                        "evaluation"
                    ),
                    notes=result_evidence.get(
                        "notes"
                    ),
                )
            )

        db.commit()
        db.refresh(version)

        return version

    @staticmethod
    def get_versions(
        db: Session,
        project_id: int,
    ):
        return (
            db.query(Version)
            .filter(
                Version.project_id == project_id
            )
            .order_by(
                Version.version_number.asc()
            )
            .all()
        )

    @staticmethod
    def get_version(
        db: Session,
        project_id: int,
        version_id: int,
    ):
        return (
            db.query(Version)
            .filter(
                Version.id == version_id,
                Version.project_id == project_id,
            )
            .first()
        )