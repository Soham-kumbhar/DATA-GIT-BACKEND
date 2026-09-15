from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.status import ProjectStatusResponse
from app.services.status_service import StatusService


router = APIRouter(
    prefix="/projects/{project_id}/status",
    tags=["Status"],
)


@router.get(
    "",
    response_model=ProjectStatusResponse,
)
def get_project_status(
    project_id: int,
    db: Session = Depends(get_db),
):
    try:
        return StatusService.get_project_status(
            db,
            project_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )