from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import Model, Project
from app.schemas.model import ModelCreate


class ModelService:

    @staticmethod
    def create_model(
        db: Session,
        project_id: int,
        model_data: ModelCreate,
    ) -> Model:

        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if project is None:
            raise ValueError("Project not found.")

        model_path = (
            Path(project.path) / model_data.path
        ).expanduser()

        if not model_path.exists():
            raise ValueError(
                "Model path does not exist."
            )

        if not model_path.is_file():
            raise ValueError(
                "Model path is not a file."
            )

        resolved_path = str(
            model_path.resolve()
        )

        existing_model = (
            db.query(Model)
            .filter(
                Model.project_id == project_id,
                Model.path == resolved_path,
            )
            .first()
        )

        if existing_model:
            raise ValueError(
                "This model is already registered."
            )

        model = Model(
            project_id=project_id,
            name=model_data.name,
            path=resolved_path,
        )

        db.add(model)
        db.commit()
        db.refresh(model)

        return model

    @staticmethod
    def get_models(
        db: Session,
        project_id: int,
    ) -> list[Model]:

        return (
            db.query(Model)
            .filter(Model.project_id == project_id)
            .all()
        )