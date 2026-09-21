from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import Dataset, Project
from app.schemas.dataset import DatasetCreate
from app.services.dvc_service import DVCService


class DatasetService:

    # ============================================================
    # GET PROJECT
    # ============================================================

    @staticmethod
    def _get_project(
        db: Session,
        project_id: int,
    ) -> Project | None:
        return (
            db.query(Project)
            .filter(
                Project.id == project_id
            )
            .first()
        )

    # ============================================================
    # CREATE DATASET
    # ============================================================

    @staticmethod
    def create_dataset(
        db: Session,
        project_id: int,
        dataset_data: DatasetCreate,
    ) -> Dataset:

        project = (
            DatasetService._get_project(
                db,
                project_id,
            )
        )

        if project is None:
            raise ValueError(
                "Project not found."
            )

        dataset_path = (
            Path(project.path)
            / dataset_data.path
        ).expanduser()

        if not dataset_path.exists():
            raise ValueError(
                "Dataset path does not exist."
            )

        if not dataset_path.is_file():
            raise ValueError(
                "Dataset path is not a file."
            )

        resolved_path = str(
            dataset_path.resolve()
        )

        existing_dataset = (
            db.query(Dataset)
            .filter(
                Dataset.project_id == project_id,
                Dataset.path == resolved_path,
            )
            .first()
        )

        if existing_dataset:
            raise ValueError(
                "This dataset is already registered."
            )

        dataset = Dataset(
            project_id=project_id,
            name=dataset_data.name.strip(),
            path=resolved_path,
        )

        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        return dataset

    # ============================================================
    # SYNC DVC DATASETS
    #
    # DVC tracked outputs are datasets belonging to the project.
    # They are synchronized into the existing Dataset table so
    # the rest of the DATAGIT dataset UI can use normal IDs.
    # ============================================================

    @staticmethod
    def sync_dvc_datasets(
        db: Session,
        project: Project,
    ) -> None:

        project_path = Path(
            project.path
        )

        if (
            not project_path.exists()
            or not project_path.is_dir()
        ):
            return

        try:
            tracked_files = (
                DVCService.get_tracked_files(
                    str(project_path)
                )
            )

        except Exception:
            return

        if not tracked_files:
            return

        changed = False

        for tracked in tracked_files:

            data_path = (
                tracked.get(
                    "data_path"
                )
            )

            if not data_path:
                continue

            # DVC paths are project-relative.
            dataset_path = (
                project_path
                / data_path
            )

            try:
                resolved_path = str(
                    dataset_path.resolve()
                )
            except OSError:
                continue

            # Keep one DATAGIT Dataset record for
            # one project + one dataset path.
            existing_dataset = (
                db.query(Dataset)
                .filter(
                    Dataset.project_id
                    == project.id,
                    Dataset.path
                    == resolved_path,
                )
                .first()
            )

            if existing_dataset:
                continue

            dataset_name = (
                Path(data_path).name
            )

            db.add(
                Dataset(
                    project_id=project.id,
                    name=dataset_name,
                    path=resolved_path,
                )
            )

            changed = True

        if changed:
            db.commit()

    # ============================================================
    # GET ALL DATASETS FOR PROJECT
    # ============================================================

    @staticmethod
    def get_datasets(
        db: Session,
        project_id: int,
    ) -> list[Dataset]:

        project = (
            DatasetService._get_project(
                db,
                project_id,
            )
        )

        if project is None:
            raise ValueError(
                "Project not found."
            )

        # Synchronize datasets that are already tracked
        # by DVC for this exact project.
        DatasetService.sync_dvc_datasets(
            db,
            project,
        )

        return (
            db.query(Dataset)
            .filter(
                Dataset.project_id
                == project_id
            )
            .order_by(
                Dataset.created_at.asc(),
                Dataset.id.asc(),
            )
            .all()
        )
