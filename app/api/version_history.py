from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.version_history import VersionHistoryResponse
from app.services.version_history_service import VersionHistoryService


router = APIRouter(
    prefix="/projects/{project_id}/versions",
    tags=["Version History"],
)


@router.get(
    "/history",
    response_model=VersionHistoryResponse,
)
def get_version_history(
    project_id: int,
    db: Session = Depends(get_db),
):
    try:
        return VersionHistoryService.get_history(
            db,
            project_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )