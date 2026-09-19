from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.version import VersionCreate, VersionResponse
from app.services.version_service import VersionService


router = APIRouter(
    prefix="/projects/{project_id}/versions",
    tags=["Versions"],
)


@router.post(
    "/finalize",
    response_model=VersionResponse,
    status_code=status.HTTP_201_CREATED,
)
def finalize_version(
    project_id: int,
    data: VersionCreate,
    db: Session = Depends(get_db),
):
    try:
        return VersionService.finalize_version(
            db=db,
            project_id=project_id,
            description=data.description,
        )

    except ValueError as error:
        message = str(error)

        if "already finalized" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=message,
            )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message,
        )


@router.get(
    "",
    response_model=list[VersionResponse],
)
def get_versions(
    project_id: int,
    db: Session = Depends(get_db),
):
    return VersionService.get_versions(
        db=db,
        project_id=project_id,
    )


@router.get(
    "/{version_id}",
    response_model=VersionResponse,
)
def get_version(
    project_id: int,
    version_id: int,
    db: Session = Depends(get_db),
):
    version = VersionService.get_version(
        db=db,
        project_id=project_id,
        version_id=version_id,
    )

    if not version:
        raise HTTPException(
            status_code=404,
            detail="Version not found.",
        )

    return version