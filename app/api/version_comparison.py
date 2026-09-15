from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.version_comparison import (
    VersionComparisonResponse,
)
from app.services.version_comparison_service import (
    VersionComparisonService,
)


router = APIRouter(
    prefix="/projects/{project_id}/versions",
    tags=["Version Comparison"],
)


# ============================================================
# SINGLE VERSION COMPARISON
# ============================================================

@router.get(
    "/compare",
    response_model=VersionComparisonResponse,
)
def compare_versions(
    project_id: int,
    version_1: int,
    version_2: int,
    db: Session = Depends(get_db),
):

    try:

        return VersionComparisonService.compare_versions(
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


# ============================================================
# MULTI-VERSION COMPARISON
# ============================================================

@router.get(
    "/multi-report",
)
def multi_version_report(
    project_id: int,
    versions: str = Query(
        ...,
        description=(
            "Comma-separated version IDs, "
            "for example: 1,2,3,4"
        ),
    ),
    mode: str = Query(
        "evolution",
        description=(
            "Comparison mode: evolution or baseline"
        ),
    ),
    db: Session = Depends(get_db),
):

    try:

        version_ids = [
            int(value.strip())
            for value in versions.split(",")
            if value.strip()
        ]

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid versions value. "
                "Use comma-separated integer IDs, "
                "for example: 1,2,3,4."
            ),
        )

    if len(version_ids) < 2:

        raise HTTPException(
            status_code=400,
            detail=(
                "At least two versions are required."
            ),
        )

    if mode not in {
        "evolution",
        "baseline",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid mode. Use 'evolution' "
                "or 'baseline'."
            ),
        )

    try:

        return VersionComparisonService.compare_multiple_versions(
            db=db,
            project_id=project_id,
            version_ids=version_ids,
            mode=mode,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )