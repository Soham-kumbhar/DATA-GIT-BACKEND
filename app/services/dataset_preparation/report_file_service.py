import json
from pathlib import Path


class DatasetPreparationReportFileService:

    REPORT_DIRECTORY = (
        Path("dataset_preparations")
        / "reports"
    )

    @staticmethod
    def save(
        report: dict,
        input_filename: str,
    ) -> dict:

        DatasetPreparationReportFileService.REPORT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        input_path = Path(input_filename)

        report_filename = (
            f"{input_path.stem}_preparation_report.json"
        )

        report_path = (
            DatasetPreparationReportFileService.REPORT_DIRECTORY
            / report_filename
        )

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "filename": report_filename,
            "format": "json",
            "size_bytes": report_path.stat().st_size,
            "path": str(report_path),
            "download_url": (
                "/dataset-preparations/reports/"
                f"{report_filename}"
            ),
        }