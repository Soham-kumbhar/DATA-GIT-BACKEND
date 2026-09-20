from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.user_models import User
from app.schemas.version import (
    VersionCreate,
    VersionResponse,
)
from app.schemas.version_report import (
    VersionReportResponse,
)
from app.services.version_service import (
    VersionService,
)
from app.services.version_report_service import (
    VersionReportService,
)


router = APIRouter(
    prefix="/projects/{project_id}/versions",
    tags=["Versions"],
)


# ============================================================
# FINALIZE VERSION
# ============================================================

@router.post(
    "/finalize",
    response_model=VersionResponse,
    status_code=status.HTTP_201_CREATED,
)
def finalize_version(
    project_id: int,
    data: VersionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    try:

        result_evidence = None

        if data.result_evidence is not None:
            result_evidence = (
                data.result_evidence.model_dump(
                    exclude_none=True
                )
            )

        return (
            VersionService.finalize_version(
                db=db,
                user_id=current_user.id,
                project_id=project_id,
                description=data.description,
                preparation_operations=(
                    data.preparation_operations
                ),
                result_evidence=result_evidence,
            )
        )

    except ValueError as error:

        message = str(error)

        if (
            "already finalized"
            in message.lower()
        ):
            raise HTTPException(
                status_code=(
                    status.HTTP_409_CONFLICT
                ),
                detail=message,
            ) from error

        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Project not found.",
        ) from error


# ============================================================
# LIST VERSIONS
# ============================================================

@router.get(
    "",
    response_model=list[VersionResponse],
)
def get_versions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    try:
        return VersionService.get_versions(
            db=db,
            user_id=current_user.id,
            project_id=project_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail="Project not found.",
        ) from error


# ============================================================
# VERSION REPORT
# ============================================================

@router.get(
    "/{version_id}/report",
    response_model=VersionReportResponse,
)
def get_version_report(
    project_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    version = (
        VersionService.get_version(
            db=db,
            user_id=current_user.id,
            project_id=project_id,
            version_id=version_id,
        )
    )

    if version is None:
        raise HTTPException(
            status_code=404,
            detail="Version not found.",
        )

    try:
        return (
            VersionReportService.build_report(
                db=db,
                project_id=project_id,
                version_id=version_id,
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


# ============================================================
# GET ONE VERSION
# ============================================================

@router.get(
    "/{version_id}",
    response_model=VersionResponse,
)
def get_version(
    project_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    version = (
        VersionService.get_version(
            db=db,
            user_id=current_user.id,
            project_id=project_id,
            version_id=version_id,
        )
    )

    if version is None:
        raise HTTPException(
            status_code=404,
            detail="Version not found.",
        )

    return version