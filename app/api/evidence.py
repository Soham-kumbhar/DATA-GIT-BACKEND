from fastapi import APIRouter, HTTPException

from app.services.evidence_service import EvidenceService


router = APIRouter(
    prefix="/evidence",
    tags=["Evidence"],
)


@router.get("/git")
def get_git_evidence(
    project_path: str,
    commit_hash: str,
):
    try:
        evidence = EvidenceService.get_project_evidence(
            project_path,
            commit_hash,
        )

        return evidence

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )