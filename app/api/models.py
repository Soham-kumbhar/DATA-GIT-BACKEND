from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.model import ModelCreate, ModelResponse
from app.services.model_service import ModelService


router = APIRouter(
    prefix="/projects/{project_id}/models",
    tags=["Models"],
)


@router.post(
    "",
    response_model=ModelResponse,
    status_code=201,
)
def create_model(
    project_id: int,
    model_data: ModelCreate,
    db: Session = Depends(get_db),
):
    try:
        return ModelService.create_model(
            db,
            project_id,
            model_data,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@router.get(
    "",
    response_model=list[ModelResponse],
)
def get_models(
    project_id: int,
    db: Session = Depends(get_db),
):
    return ModelService.get_models(
        db,
        project_id,
    )