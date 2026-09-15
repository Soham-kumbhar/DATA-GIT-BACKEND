from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.run_comparison_service import RunComparisonService
from app.services.smart_explanation_service import SmartExplanationService


router = APIRouter(
    prefix="/projects",
    tags=["Run Comparison"],
)


@router.get("/{project_id}/runs/compare")
def compare_runs(
    project_id: int,
    run_1: int,
    run_2: int,
    db: Session = Depends(get_db),
):
    try:
        # ---------------------------------------------
        # 1. Compare the two ML runs
        # ---------------------------------------------
        result = RunComparisonService.compare_runs(
            db=db,
            run_id_1=run_1,
            run_id_2=run_2,
        )

        # ---------------------------------------------
        # 2. Get code analysis if available
        # ---------------------------------------------
        code_analysis = result.get(
            "code_analysis"
        )

        # ---------------------------------------------
        # 3. Get dataset diff if available
        # ---------------------------------------------
        dataset_diff = result.get(
            "dataset_diff"
        )

        # ---------------------------------------------
        # 4. Generate smart explanation
        # ---------------------------------------------
        explanation = SmartExplanationService.generate(
            comparison=result,
            code_analysis=code_analysis,
            dataset_diff=dataset_diff,
        )

        # ---------------------------------------------
        # 5. Add explanation to response
        # ---------------------------------------------
        result["explanation"] = explanation

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )