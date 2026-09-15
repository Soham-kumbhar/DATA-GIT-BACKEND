from pathlib import Path

from fastapi import UploadFile


ALLOWED_EXTENSIONS = {".csv"}

UPLOAD_DIR = Path("dataset_preparations") / "uploads"


class DatasetUploadService:
    @staticmethod
    async def save_upload(file: UploadFile) -> dict:
        if not file.filename:
            raise ValueError("No file name was provided.")

        filename = Path(file.filename).name
        extension = Path(filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(
                "Unsupported file format. "
                "Dataset Preparation currently supports CSV files."
            )

        content = await file.read()

        if not content:
            raise ValueError("The uploaded file is empty.")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        destination = UPLOAD_DIR / filename
        destination.write_bytes(content)

        return {
            "filename": filename,
            "file_type": "CSV",
            "size_bytes": len(content),
            "status": "uploaded",
            "message": "Dataset uploaded successfully.",
        }