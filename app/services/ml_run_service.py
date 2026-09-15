from sqlalchemy.orm import Session

from app.db.models import MLRun, Project, Version
from app.services.dvc_service import DVCService
from app.services.git_service import GitService


class MLRunService:

    @staticmethod
    def create_run(
        db: Session,
        project_id: int,
        model_name: str,
        features: list[str] | None = None,
        parameters: dict | None = None,
        metrics: dict | None = None,
        evaluation: dict | None = None,
    ) -> MLRun:
        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if not project:
            raise ValueError("Project not found")

        git_commit = GitService.get_current_commit(project.path)
        dvc_state = DVCService.get_state(project.path)

        ml_run = MLRun(
            project_id=project_id,
            git_commit=git_commit,
            dvc_state=dvc_state,
            model_name=model_name,
            features=features,
            parameters=parameters,
            metrics=metrics,
            evaluation=evaluation,
        )

        db.add(ml_run)
        db.commit()
        db.refresh(ml_run)

        return ml_run

    @staticmethod
    def create_run_for_version(
        db: Session,
        project_id: int,
        version_id: int,
        model_name: str,
        features: list[str] | None = None,
        parameters: dict | None = None,
        metrics: dict | None = None,
        evaluation: dict | None = None,
    ) -> MLRun:
        version = (
            db.query(Version)
            .filter(
                Version.id == version_id,
                Version.project_id == project_id,
            )
            .first()
        )

        if not version:
            raise ValueError("Version not found")

        ml_run = MLRun(
            project_id=project_id,
            git_commit=version.git_commit,
            dvc_state=version.dvc_state,
            model_name=model_name,
            features=features,
            parameters=parameters,
            metrics=metrics,
            evaluation=evaluation,
        )

        db.add(ml_run)
        db.flush()

        version.ml_run_id = ml_run.id

        db.commit()
        db.refresh(ml_run)

        return ml_run

    @staticmethod
    def get_runs(
        db: Session,
        project_id: int,
    ) -> list[MLRun]:
        return (
            db.query(MLRun)
            .filter(MLRun.project_id == project_id)
            .order_by(MLRun.created_at.desc())
            .all()
        )