from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.deep_compare_service import (
    DeepCompareService,
)

router = APIRouter(
    tags=["Compare"],
)


@router.get(
    "/projects/{project_id}/compare",
)
def compare_versions(
    project_id: int,
    version_a: int = Query(
        ...,
        description="Baseline version database id.",
    ),
    version_b: int = Query(
        ...,
        description="Target version database id.",
    ),
    generate_ai: bool = Query(
        True,
        description=(
            "Generate the evidence-grounded AI comparison "
            "after deterministic analysis."
        ),
    ),
    db: Session = Depends(get_db),
):
    if version_a == version_b:
        raise HTTPException(
            status_code=422,
            detail=(
                "Version A and Version B must be different."
            ),
        )

    try:
        return DeepCompareService.compare(
            db=db,
            project_id=project_id,
            version_a_id=version_a,
            version_b_id=version_b,
            generate_ai=generate_ai,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Compare analysis failed. "
                f"{error}"
            ),
        ) from error