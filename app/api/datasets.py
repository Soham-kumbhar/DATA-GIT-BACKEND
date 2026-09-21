from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.user_models import User
from app.db.models import Project
from app.schemas.dataset import (
    DatasetCreate,
    DatasetResponse,
)
from app.services.dataset_service import (
    DatasetService,
)


router = APIRouter(
    prefix="/projects/{project_id}/datasets",
    tags=["Datasets"],
)


# ============================================================
# CREATE DATASET
# ============================================================

@router.post(
    "",
    response_model=DatasetResponse,
    status_code=201,
)
def create_dataset(
    project_id: int,
    dataset_data: DatasetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    project = (
        db.query(Project)
        .filter(
            Project.id == project_id,
            Project.user_id == current_user.id,
        )
        .first()
    )

    if project is None:
        raise HTTPException(
            status_code=404,
            detail="Project not found.",
        )

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
        ) from error


# ============================================================
# GET ALL DATASETS FOR PROJECT
# ============================================================

@router.get(
    "",
    response_model=list[DatasetResponse],
)
def get_datasets(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    # Never allow one user to read another user's
    # project datasets.
    project = (
        db.query(Project)
        .filter(
            Project.id == project_id,
            Project.user_id == current_user.id,
        )
        .first()
    )

    if project is None:
        raise HTTPException(
            status_code=404,
            detail="Project not found.",
        )

    try:
        return DatasetService.get_datasets(
            db,
            project_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
