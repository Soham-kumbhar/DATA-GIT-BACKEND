from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.dataset import DatasetCreate, DatasetResponse
from app.services.dataset_service import DatasetService


router = APIRouter(
    prefix="/projects/{project_id}/datasets",
    tags=["Datasets"],
)


@router.post(
    "",
    response_model=DatasetResponse,
    status_code=201,
)
def create_dataset(
    project_id: int,
    dataset_data: DatasetCreate,
    db: Session = Depends(get_db),
):
    try:
        return DatasetService.create_dataset(
            db,
            project_id,
            dataset_data,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@router.get(
    "",
    response_model=list[DatasetResponse],
)
def get_datasets(
    project_id: int,
    db: Session = Depends(get_db),
):
    return DatasetService.get_datasets(
        db,
        project_id,
    )