from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from app.schemas.dataset_preparation import (
    PreparationConfigurationRequest,
    PreparationConfigurationResponse,
    PreparationMissingValuesRequest,
    PreparationProceedRequest,
    PreparationProcessRequest,
    PreparationReportRequest,
    PreparationSelectionRequest,
    PreparationSelectionResponse,
)

from app.services.dataset_preparation.handle_missing_service import (
    HandleMissingService,
)

from app.services.dataset_preparation.profile_service import (
    DatasetProfileService,
)

from app.services.dataset_preparation.process_service import (
    DatasetPreparationProcessService,
)

from app.services.dataset_preparation.remove_duplicates_service import (
    RemoveDuplicatesService,
)

from app.services.dataset_preparation.report_file_service import (
    DatasetPreparationReportFileService,
)

from app.services.dataset_preparation.report_service import (
    DatasetPreparationReportService,
)

from app.services.dataset_preparation.upload_service import (
    DatasetUploadService,
)

from app.services.dataset_preparation.validation_service import (
    DatasetPreparationValidationService,
)

from app.services.groq_report_service import (
    GroqReportService,
)


router = APIRouter(
    prefix="/dataset-preparations",
    tags=["Dataset Preparations"],
)


# ============================================================
# PREPARATION CATEGORIES + OPERATIONS
# ============================================================

PREPARATION_OPERATION_CATEGORIES = {
    "Handling Missing Values": [
        {
            "operation": "handle_missing",
            "status": "available",
            "description": (
                "Handle missing values using a selected strategy."
            ),
            "strategies": [
                "drop_rows",
                "drop_column",
                "mean",
                "median",
                "mode",
                "constant",
                "forward_fill",
                "backward_fill",
            ],
        },
        {
            "operation": "knn_imputation",
            "status": "available",
            "description": (
                "Fill missing numeric values using "
                "K-Nearest Neighbors imputation."
            ),
            "weights": [
                "uniform",
                "distance",
            ],
            "default_n_neighbors": 5,
        },
        {
            "operation": "mice_imputation",
            "status": "available",
            "description": (
                "Fill missing numeric values using "
                "Multiple Imputation by Chained Equations."
            ),
            "default_max_iter": 10,
            "default_random_state": 42,
        },
    ],
    "Duplicates & Irrelevant Data": [
        {
            "operation": "remove_duplicates",
            "status": "available",
            "description": (
                "Remove exact or partial duplicates."
            ),
        },
        {
            "operation": "drop_columns",
            "status": "available",
            "description": (
                "Remove user-selected columns."
            ),
        },
    ],
    "Managing Outliers": [
        {
            "operation": "trim_outliers",
            "status": "available",
            "description": (
                "Remove rows containing selected outliers."
            ),
        },
        {
            "operation": "winsorize_outliers",
            "status": "available",
            "description": (
                "Cap selected outlier values."
            ),
        },
        {
            "operation": "log_transform",
            "status": "available",
            "description": (
                "Apply a logarithmic transformation "
                "to selected numeric columns."
            ),
        },
        {
            "operation": "box_cox_transform",
            "status": "available",
            "description": (
                "Apply a Box-Cox transformation "
                "to selected positive numeric columns."
            ),
        },
    ],
    "Structural / Format Standardization": [
        {
            "operation": "type_conversion",
            "status": "available",
            "description": (
                "Convert selected columns to a target data type."
            ),
            "target_types": [
                "integer",
                "float",
                "string",
                "boolean",
                "date",
                "datetime",
            ],
        },
        {
            "operation": "string_standardization",
            "status": "available",
            "description": (
                "Standardize selected string columns."
            ),
            "options": [
                "trim_whitespace",
                "lowercase",
                "uppercase",
                "remove_extra_spaces",
            ],
        },
        {
            "operation": "categorical_uniformity",
            "status": "available",
            "description": (
                "Standardize inconsistent categorical values "
                "using explicit mappings."
            ),
        },
        {
            "operation": "schema_alignment",
            "status": "available",
            "description": (
                "Align dataset columns to a target schema."
            ),
        },
    ],
    "Inconsistent / Erroneous Data": [
        {
            "operation": "logical_rule_validation",
            "status": "available",
            "description": (
                "Validate dataset rows against user-defined "
                "logical rules without modifying the dataset."
            ),
        },
        {
            "operation": "syntax_correction",
            "status": "available",
            "description": (
                "Correct common syntax and formatting errors "
                "in selected columns."
            ),
            "correction_types": [
                "email",
                "phone",
                "numeric",
                "date",
                "datetime",
                "whitespace",
                "alphanumeric",
            ],
        },
        {
            "operation": "dummy_value_replacement",
            "status": "available",
            "description": (
                "Replace explicitly configured dummy values "
                "in selected columns."
            ),
        },
    ],
    "Feature Scaling / Transformation": [
        {
            "operation": "min_max_scaling",
            "status": "available",
            "description": (
                "Scale selected numeric columns to a configured range "
                "using Min-Max normalization."
            ),
            "default_feature_range": [
                0,
                1,
            ],
        },
        {
            "operation": "z_score_scaling",
            "status": "available",
            "description": (
                "Standardize selected numeric columns "
                "using Z-score scaling."
            ),
        },
        {
            "operation": "binning",
            "status": "available",
            "description": (
                "Divide selected numeric columns into discrete bins."
            ),
            "methods": [
                "equal_width",
                "quantile",
            ],
        },
    ],
    "Imbalanced Data": [
        {
            "operation": "oversampling",
            "status": "available",
            "description": (
                "Balance class distribution by randomly "
                "oversampling minority classes."
            ),
        },
        {
            "operation": "smote",
            "status": "available",
            "description": (
                "Balance class distribution using "
                "Synthetic Minority Over-sampling Technique."
            ),
        },
        {
            "operation": "undersampling",
            "status": "available",
            "description": (
                "Balance class distribution by randomly "
                "undersampling majority classes."
            ),
        },
    ],
}


# ============================================================
# CURRENTLY SUPPORTED OPERATIONS
# ============================================================

AVAILABLE_OPERATIONS = [
    "remove_duplicates",
    "handle_missing",
    "drop_columns",
    "trim_outliers",
    "winsorize_outliers",
    "log_transform",
    "box_cox_transform",
    "type_conversion",
    "string_standardization",
    "categorical_uniformity",
    "schema_alignment",
    "logical_rule_validation",
    "syntax_correction",
    "dummy_value_replacement",
    "min_max_scaling",
    "z_score_scaling",
    "binning",
    "oversampling",
    "smote",
    "undersampling",
    "knn_imputation",
    "mice_imputation",
]


# ============================================================
# LIST PREPARATION CATEGORIES + OPERATIONS
# ============================================================

@router.get("/operations")
def list_preparation_operations():
    return {
        "status": "success",
        "categories": PREPARATION_OPERATION_CATEGORIES,
    }


# ============================================================
# UPLOAD DATASET
# ============================================================

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        return await DatasetUploadService.save_upload(file)

    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(error),
            },
        )


# ============================================================
# PROFILE DATASET
# ============================================================

@router.get("/profile/{filename}")
def profile_dataset(filename: str):
    try:
        safe_filename = Path(filename).name

        file_path = (
            Path("dataset_preparations")
            / "uploads"
            / safe_filename
        )

        return DatasetProfileService.profile(
            str(file_path)
        )

    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(error),
            },
        )


# ============================================================
# SELECT PREPARATION OPERATIONS
# ============================================================

@router.post(
    "/select",
    response_model=PreparationSelectionResponse,
)
def select_preparation_operations(
    request: PreparationSelectionRequest,
):
    if not request.operations:
        raise HTTPException(
            status_code=400,
            detail="Select at least one preparation operation.",
        )

    invalid_operations = [
        operation
        for operation in request.operations
        if operation not in AVAILABLE_OPERATIONS
    ]

    if invalid_operations:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "One or more selected operations "
                    "are not currently supported."
                ),
                "invalid_operations": invalid_operations,
                "available_operations": AVAILABLE_OPERATIONS,
            },
        )

    return PreparationSelectionResponse(
        filename=request.filename,
        selected_operations=request.operations,
        operation_count=len(request.operations),
        status="selected",
        message=(
            "Preparation operations selected successfully. "
            "Configure the selected operations and click Proceed "
            "to execute them."
        ),
    )


# ============================================================
# CONFIGURE PREPARATION OPERATIONS
# ============================================================

@router.post(
    "/configure",
    response_model=PreparationConfigurationResponse,
)
def configure_preparation_operations(
    request: PreparationConfigurationRequest,
):
    if not request.operations:
        raise HTTPException(
            status_code=400,
            detail="Configure at least one preparation operation.",
        )

    configured_operations = []

    for operation in request.operations:
        name = operation.get("operation")

        if name not in AVAILABLE_OPERATIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported operation: {name}",
            )

        configured_operations.append(
            operation
        )

    return PreparationConfigurationResponse(
        filename=request.filename,
        operations=configured_operations,
        operation_count=len(configured_operations),
        status="configured",
        message=(
            "Preparation operations configured successfully. "
            "Click Proceed to execute the complete plan."
        ),
    )


# ============================================================
# VALIDATE PREPARATION CONFIGURATION
# ============================================================

@router.post("/validate")
def validate_preparation_configuration(
    request: PreparationConfigurationRequest,
):
    safe_filename = Path(
        request.filename
    ).name

    file_path = (
        Path("dataset_preparations")
        / "uploads"
        / safe_filename
    )

    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Uploaded dataset was not found.",
            },
        )

    try:
        result = (
            DatasetPreparationValidationService.validate(
                file_path=str(file_path),
                operations=request.operations,
            )
        )

        return {
            "filename": safe_filename,
            **result,
        }

    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(error),
            },
        )


# ============================================================
# PROCEED
#
# User selects + configures operations first.
# Nothing is processed until this endpoint is called.
#
# Validation happens BEFORE processing.
# If validation fails, processing DOES NOT happen.
# ============================================================

@router.post("/proceed")
def proceed_with_dataset_preparation(
    request: PreparationProceedRequest,
):
    safe_filename = Path(
        request.filename
    ).name

    input_path = (
        Path("dataset_preparations")
        / "uploads"
        / safe_filename
    )

    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded dataset was not found.",
        )

    if not request.operations:
        raise HTTPException(
            status_code=400,
            detail="No preparation operations were selected.",
        )

    invalid_operations = [
        operation.get("operation")
        for operation in request.operations
        if operation.get("operation")
        not in AVAILABLE_OPERATIONS
    ]

    if invalid_operations:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "One or more selected operations "
                    "are not currently supported."
                ),
                "invalid_operations": invalid_operations,
                "available_operations": AVAILABLE_OPERATIONS,
            },
        )

    try:
        validation_result = (
            DatasetPreparationValidationService.validate(
                file_path=str(input_path),
                operations=request.operations,
            )
        )

        if not validation_result.get(
            "valid",
            False,
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failed",
                    "message": (
                        "Dataset preparation validation failed. "
                        "No operations were executed."
                    ),
                    "filename": safe_filename,
                    "operation_count": len(
                        request.operations
                    ),
                    "selected_operations": request.operations,
                    "validation": validation_result,
                },
            )

        process_result = (
            DatasetPreparationProcessService.execute(
                file_path=str(input_path),
                operations=request.operations,
                output_format=request.output_format,
            )
        )

        return {
            "status": "success",
            "message": (
                "All selected preparation operations "
                "were validated and executed successfully."
            ),
            "filename": safe_filename,
            "operation_count": len(
                request.operations
            ),
            "selected_operations": request.operations,
            "validation": validation_result,
            "result": process_result,
        }

    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "message": (
                    "Dataset preparation could not proceed."
                ),
                "detail": str(error),
            },
        )

    except Exception as error:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "message": "Dataset preparation failed.",
                "detail": str(error),
            },
        )


# ============================================================
# PROCESS: REMOVE DUPLICATES
# ============================================================

@router.post(
    "/process/remove-duplicates"
)
def process_remove_duplicates(
    filename: str,
    strategy: str = "keep_first",
):
    safe_filename = Path(
        filename
    ).name

    input_path = (
        Path("dataset_preparations")
        / "uploads"
        / safe_filename
    )

    try:
        return RemoveDuplicatesService.execute(
            file_path=str(input_path),
            strategy=strategy,
        )

    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(error),
            },
        )


# ============================================================
# PROCESS: HANDLE MISSING VALUES
# ============================================================

@router.post(
    "/process/handle-missing"
)
def process_handle_missing(
    request: PreparationMissingValuesRequest,
):
    safe_filename = Path(
        request.filename
    ).name

    input_path = (
        Path("dataset_preparations")
        / "uploads"
        / safe_filename
    )

    try:
        return HandleMissingService.execute(
            file_path=str(input_path),
            columns=request.columns,
            constant_values=request.constant_values,
        )

    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(error),
            },
        )


# ============================================================
# PROCESS COMPLETE DATASET PREPARATION
# ============================================================

@router.post("/process")
def process_dataset_preparation(
    request: PreparationProcessRequest,
):
    safe_filename = Path(
        request.filename
    ).name

    input_path = (
        Path("dataset_preparations")
        / "uploads"
        / safe_filename
    )

    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded dataset was not found.",
        )

    try:
        return DatasetPreparationProcessService.execute(
            file_path=str(input_path),
            operations=request.operations,
            output_format=request.output_format,
        )

    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(error),
            },
        )


# ============================================================
# GENERATE AND SAVE DATASET PREPARATION REPORT
# ============================================================

@router.post("/report")
def generate_preparation_report(
    request: PreparationReportRequest,
):
    try:
        deterministic_report = (
            DatasetPreparationReportService.generate(
                input_file=request.filename,
                output_file=request.output_file,
                operations=request.operations,
            )
        )

        ai_insights = (
            GroqReportService.analyze_dataset_preparation(
                deterministic_report
            )
        )

        complete_report = {
            **deterministic_report,
            "ai_insights": ai_insights,
        }

        report_file = (
            DatasetPreparationReportFileService.save(
                report=complete_report,
                input_filename=request.filename,
            )
        )

        return {
            **complete_report,
            "report_file": report_file,
        }

    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(error),
            },
        )

    except Exception as error:
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(error),
            },
        )


# ============================================================
# LIST PREPARED DATASETS
# ============================================================

@router.get("/prepared")
def list_prepared_datasets():
    prepared_directory = (
        Path("dataset_preparations")
        / "prepared"
    )

    if not prepared_directory.exists():
        return {
            "files": []
        }

    files = []

    for file_path in sorted(
        prepared_directory.iterdir()
    ):
        if not file_path.is_file():
            continue

        files.append(
            {
                "filename": file_path.name,
                "format": (
                    file_path
                    .suffix
                    .lower()
                    .lstrip(".")
                ),
                "size_bytes": (
                    file_path.stat().st_size
                ),
                "download_url": (
                    "/dataset-preparations/"
                    "prepared/"
                    f"{file_path.name}"
                ),
            }
        )

    return {
        "files": files
    }


# ============================================================
# LIST DATASET PREPARATION REPORTS
# ============================================================

@router.get("/reports")
def list_preparation_reports():
    reports_directory = (
        Path("dataset_preparations")
        / "reports"
    )

    if not reports_directory.exists():
        return {
            "files": []
        }

    files = []

    for file_path in sorted(
        reports_directory.iterdir()
    ):
        if not file_path.is_file():
            continue

        files.append(
            {
                "filename": file_path.name,
                "format": (
                    file_path
                    .suffix
                    .lower()
                    .lstrip(".")
                ),
                "size_bytes": (
                    file_path.stat().st_size
                ),
                "download_url": (
                    "/dataset-preparations/"
                    "reports/"
                    f"{file_path.name}"
                ),
            }
        )

    return {
        "files": files
    }


# ============================================================
# DOWNLOAD PREPARED DATA
# ============================================================

@router.get(
    "/prepared/{filename}"
)
def download_prepared_dataset(
    filename: str,
):
    safe_filename = Path(
        filename
    ).name

    prepared_path = (
        Path("dataset_preparations")
        / "prepared"
        / safe_filename
    )

    if not prepared_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Prepared dataset was not found.",
        )

    return FileResponse(
        path=prepared_path,
        filename=safe_filename,
        media_type="application/octet-stream",
    )


# ============================================================
# DOWNLOAD DATASET PREPARATION REPORT
# ============================================================

@router.get(
    "/reports/{filename}"
)
def download_preparation_report(
    filename: str,
):
    safe_filename = Path(
        filename
    ).name

    report_path = (
        Path("dataset_preparations")
        / "reports"
        / safe_filename
    )

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Dataset preparation report "
                "was not found."
            ),
        )

    return FileResponse(
        path=report_path,
        filename=safe_filename,
        media_type="application/json",
    )