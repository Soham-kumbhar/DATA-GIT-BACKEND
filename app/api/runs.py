from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Project
from app.schemas.ml_run import MLRunCreate, MLRunResponse
from app.services.ml_run_service import MLRunService


router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["ML Runs"],
)


@router.post(
    "/runs",
    response_model=MLRunResponse,
    status_code=201,
)
def create_ml_run(
    project_id: int,
    data: MLRunCreate,
    db: Session = Depends(get_db),
):
    project = (
        db.query(Project)
        .filter(Project.id == project_id)
        .first()
    )

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found",
        )

    try:
        run = MLRunService.create_run(
            db=db,
            project_id=project_id,
            model_name=data.model_name,
            features=data.features,
            parameters=data.parameters,
            metrics=data.metrics,
            evaluation=data.evaluation,
        )

        return run

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@router.post(
    "/versions/{version_id}/runs",
    response_model=MLRunResponse,
    status_code=201,
)
def create_ml_run_for_version(
    project_id: int,
    version_id: int,
    data: MLRunCreate,
    db: Session = Depends(get_db),
):
    project = (
        db.query(Project)
        .filter(Project.id == project_id)
        .first()
    )

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found",
        )

    try:
        run = MLRunService.create_run_for_version(
            db=db,
            project_id=project_id,
            version_id=version_id,
            model_name=data.model_name,
            features=data.features,
            parameters=data.parameters,
            metrics=data.metrics,
            evaluation=data.evaluation,
        )

        return run

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@router.get(
    "/runs",
    response_model=list[MLRunResponse],
)
def get_ml_runs(
    project_id: int,
    db: Session = Depends(get_db),
):
    project = (
        db.query(Project)
        .filter(Project.id == project_id)
        .first()
    )

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found",
        )

    return MLRunService.get_runs(
        db=db,
        project_id=project_id,
    )