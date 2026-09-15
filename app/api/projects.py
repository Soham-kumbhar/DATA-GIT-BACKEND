from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.project import ProjectCreate, ProjectResponse
from app.services.project_service import ProjectService


router = APIRouter(
    prefix="/projects",
    tags=["Projects"],
)


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=201,
)
def create_project(
    project_data: ProjectCreate,
    db: Session = Depends(get_db),
):
    try:
        return ProjectService.create_project(
            db,
            project_data,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@router.get(
    "",
    response_model=list[ProjectResponse],
)
def get_projects(
    db: Session = Depends(get_db),
):
    return ProjectService.get_projects(db)


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
)
def get_project(
    project_id: int,
    db: Session = Depends(get_db),
):
    project = ProjectService.get_project(
        db,
        project_id,
    )

    if project is None:
        raise HTTPException(
            status_code=404,
            detail="Project not found.",
        )

    return project