from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.version_report import (
    VersionReportResponse,
)
from app.services.version_report_service import (
    VersionReportService,
)


router = APIRouter(
    prefix="/projects/{project_id}/versions",
    tags=["Version Report"],
)


@router.get(
    "/report",
    response_model=VersionReportResponse,
)
def get_version_report(
    project_id: int,
    version_1: int,
    version_2: int,
    db: Session = Depends(get_db),
):

    try:

        return VersionReportService.build_report(
            db=db,
            project_id=project_id,
            version_1=version_1,
            version_2=version_2,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except RuntimeError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )