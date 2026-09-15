from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import Dataset, Project
from app.schemas.dataset import DatasetCreate


class DatasetService:

    @staticmethod
    def create_dataset(
        db: Session,
        project_id: int,
        dataset_data: DatasetCreate,
    ) -> Dataset:

        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if project is None:
            raise ValueError("Project not found.")

        dataset_path = (
            Path(project.path) / dataset_data.path
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
            name=dataset_data.name,
            path=resolved_path,
        )

        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        return dataset

    @staticmethod
    def get_datasets(
        db: Session,
        project_id: int,
    ) -> list[Dataset]:

        return (
            db.query(Dataset)
            .filter(Dataset.project_id == project_id)
            .all()
        )