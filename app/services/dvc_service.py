from pathlib import Path
import subprocess
import tempfile
import sys

import pandas as pd


class DVCService:

    # ============================================================
    # RUN DVC COMMAND
    # ============================================================

    @staticmethod
    def _run_dvc(
        project_path: str,
        *args: str,
    ) -> str:

        result = subprocess.run(
            [sys.executable, "-m", "dvc", *args],
            cwd=project_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
            )

        return result.stdout.strip()

    # ============================================================
    # CHECK DVC REPOSITORY
    # ============================================================

    @staticmethod
    def is_dvc_repository(
        project_path: str,
    ) -> bool:

        path = Path(project_path)

        if not path.exists() or not path.is_dir():
            return False

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dvc",
                "root",
            ],
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        return result.returncode == 0

    # ============================================================
    # DVC STATUS
    # ============================================================

    @staticmethod
    def get_status(
        project_path: str,
    ) -> str:

        return DVCService._run_dvc(
            project_path,
            "status",
        )

    # ============================================================
    # DVC DIFF
    # ============================================================

    @staticmethod
    def get_diff(
        project_path: str,
    ) -> str:

        return DVCService._run_dvc(
            project_path,
            "diff",
        )

    # ============================================================
    # GET DVC TRACKED FILES
    # ============================================================

    @staticmethod
    def get_tracked_files(
        project_path: str,
    ) -> list[dict]:

        project = Path(project_path)

        if not DVCService.is_dvc_repository(
            project_path
        ):
            return []

        tracked_files = []

        for dvc_file in project.rglob("*.dvc"):

            try:

                lines = dvc_file.read_text(
                    encoding="utf-8"
                ).splitlines()

                md5 = None
                size = None
                data_path = None

                for line in lines:

                    line = line.strip()

                    # DVC format:
                    # - md5: abc123
                    if line.startswith("- md5:"):

                        md5 = line.split(
                            "md5:",
                            1,
                        )[1].strip()

                    # Also support:
                    # md5: abc123
                    elif line.startswith("md5:"):

                        md5 = line.split(
                            "md5:",
                            1,
                        )[1].strip()

                    elif line.startswith("size:"):

                        size = int(
                            line.split(
                                ":",
                                1,
                            )[1].strip()
                        )

                    elif line.startswith("path:"):

                        data_path = line.split(
                            ":",
                            1,
                        )[1].strip()

                tracked_files.append(
                    {
                        "dvc_file": str(
                            dvc_file.relative_to(
                                project
                            )
                        ),
                        "data_path": data_path,
                        "md5": md5,
                        "size": size,
                    }
                )

            except (
                OSError,
                UnicodeDecodeError,
                ValueError,
            ):
                continue

        return tracked_files

    # ============================================================
    # COMPLETE DVC STATE
    # ============================================================

    @staticmethod
    def get_state(
        project_path: str,
    ) -> dict:

        return {
            "is_repository":
                DVCService.is_dvc_repository(
                    project_path
                ),

            "status":
                DVCService.get_status(
                    project_path
                ),

            "diff":
                DVCService.get_diff(
                    project_path
                ),

            "tracked_files":
                DVCService.get_tracked_files(
                    project_path
                ),
        }

    # ============================================================
    # EXTRACT DVC HASH
    # ============================================================

    @staticmethod
    def _extract_dvc_hash(
        dvc_content: str,
    ) -> str | None:

        """
        Extract MD5 hash from DVC metadata.

        Supports:

            - md5: abc123

        and:

            md5: abc123
        """

        for line in dvc_content.splitlines():

            line = line.strip()

            if line.startswith("- md5:"):

                return line.split(
                    "md5:",
                    1,
                )[1].strip()

            if line.startswith("md5:"):

                return line.split(
                    "md5:",
                    1,
                )[1].strip()

        return None

    # ============================================================
    # GET DVC METADATA FROM GIT COMMIT
    # ============================================================

    @staticmethod
    def _get_dvc_metadata(
        project_path: str,
        commit: str,
        dvc_file_path: str,
    ) -> str:

        """
        Get the .dvc metadata file from a
        particular Git commit.
        """

        dvc_file_path = dvc_file_path.replace(
            "\\",
            "/",
        )

        result = subprocess.run(
            [
                "git",
                "show",
                f"{commit}:{dvc_file_path}",
            ],
            cwd=project_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "Could not retrieve DVC metadata "
                f"from commit {commit}.\n"
                f"DVC file: {dvc_file_path}\n"
                f"Git error: {result.stderr.strip()}"
            )

        return result.stdout

    # ============================================================
    # GET DVC CACHE DIRECTORY
    # ============================================================

    @staticmethod
    def get_cache_dir(
        project_path: str,
    ) -> str:

        return DVCService._run_dvc(
            project_path,
            "cache",
            "dir",
        )

    # ============================================================
    # GET DATASET FROM DVC CACHE
    # ============================================================

    @staticmethod
    def _get_from_cache(
        project_path: str,
        dvc_hash: str,
        output_path: Path,
    ) -> None:

        """
        Retrieve a dataset from the local DVC cache.

        Standard DVC cache structure:

        cache/
        └── files/
            └── md5/
                └── f8/
                    └── 2891ea...
        """

        cache_dir_raw = DVCService.get_cache_dir(
            project_path
        )

        cache_dir = Path(
            cache_dir_raw.strip().strip('"')
        )

        # --------------------------------------------------------
        # Standard DVC cache structure
        # --------------------------------------------------------

        cache_file = (
            cache_dir
            / "files"
            / "md5"
            / dvc_hash[:2]
            / dvc_hash[2:]
        )

        # --------------------------------------------------------
        # Alternative cache structure
        # --------------------------------------------------------

        if not cache_file.exists():

            alternative_cache_file = (
                cache_dir
                / dvc_hash[:2]
                / dvc_hash[2:]
            )

            if alternative_cache_file.exists():

                cache_file = (
                    alternative_cache_file
                )

        # --------------------------------------------------------
        # Cache object not found
        # --------------------------------------------------------

        if not cache_file.exists():

            raise RuntimeError(
                "DVC object was not found in "
                "the local cache.\n\n"
                f"Hash: {dvc_hash}\n"
                f"Cache directory: {cache_dir}\n"
                f"Expected object: {cache_file}"
            )

        # --------------------------------------------------------
        # Copy dataset
        # --------------------------------------------------------

        output_path.write_bytes(
            cache_file.read_bytes()
        )

    # ============================================================
    # DATASET DIFF
    # ============================================================

    @staticmethod
    def get_dataset_diff(
        project_path: str,
        old_commit: str,
        new_commit: str,
        data_path: str,
    ) -> dict:

        """
        Compare the actual dataset between
        two Git/DVC versions.

        Example:

        OLD COMMIT
            |
            v
        cars.csv.dvc
            |
            v
        old MD5
            |
            v
        OLD DATASET

        NEW COMMIT
            |
            v
        cars.csv.dvc
            |
            v
        new MD5
            |
            v
        NEW DATASET

        Then compare the two datasets.
        """

        # ========================================================
        # NORMALIZE PATH
        # ========================================================

        data_path = data_path.replace(
            "\\",
            "/",
        ).lstrip("./")

        # If caller gives:
        #
        # data/cars.csv
        #
        # DVC file is:
        #
        # data/cars.csv.dvc

        if data_path.endswith(".dvc"):

            dvc_file_path = data_path

            actual_data_path = data_path[
                :-4
            ]

        else:

            dvc_file_path = (
                data_path + ".dvc"
            )

            actual_data_path = data_path

        # ========================================================
        # GET OLD DVC METADATA
        # ========================================================

        old_dvc_content = (
            DVCService._get_dvc_metadata(
                project_path=project_path,
                commit=old_commit,
                dvc_file_path=dvc_file_path,
            )
        )

        # ========================================================
        # GET NEW DVC METADATA
        # ========================================================

        new_dvc_content = (
            DVCService._get_dvc_metadata(
                project_path=project_path,
                commit=new_commit,
                dvc_file_path=dvc_file_path,
            )
        )

        # ========================================================
        # EXTRACT OLD HASH
        # ========================================================

        old_hash = (
            DVCService._extract_dvc_hash(
                old_dvc_content
            )
        )

        # ========================================================
        # EXTRACT NEW HASH
        # ========================================================

        new_hash = (
            DVCService._extract_dvc_hash(
                new_dvc_content
            )
        )

        # ========================================================
        # VALIDATE OLD HASH
        # ========================================================

        if old_hash is None:

            raise RuntimeError(
                "Could not find old DVC hash.\n"
                f"Commit: {old_commit}\n"
                f"DVC file: {dvc_file_path}\n\n"
                f"DVC metadata:\n{old_dvc_content}"
            )

        # ========================================================
        # VALIDATE NEW HASH
        # ========================================================

        if new_hash is None:

            raise RuntimeError(
                "Could not find new DVC hash.\n"
                f"Commit: {new_commit}\n"
                f"DVC file: {dvc_file_path}\n\n"
                f"DVC metadata:\n{new_dvc_content}"
            )

        # ========================================================
        # DATASET DID NOT CHANGE
        # ========================================================

        if old_hash == new_hash:

            return {
                "dataset": actual_data_path,

                "dataset_changed": False,

                "old_dvc_hash": old_hash,

                "new_dvc_hash": new_hash,

                "old_rows": None,

                "new_rows": None,

                "old_columns": [],

                "new_columns": [],

                "columns_added": [],

                "columns_removed": [],

                "rows_added": 0,

                "rows_removed": 0,

                "added_rows": [],

                "removed_rows": [],
            }

        # ========================================================
        # CREATE TEMPORARY DIRECTORY
        # ========================================================

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_path = Path(
                temp_dir
            )

            old_dataset = (
                temp_path
                / "old_dataset.csv"
            )

            new_dataset = (
                temp_path
                / "new_dataset.csv"
            )

            # ====================================================
            # GET OLD DATASET
            # ====================================================

            DVCService._get_from_cache(
                project_path=project_path,
                dvc_hash=old_hash,
                output_path=old_dataset,
            )

            # ====================================================
            # GET NEW DATASET
            # ====================================================

            DVCService._get_from_cache(
                project_path=project_path,
                dvc_hash=new_hash,
                output_path=new_dataset,
            )

            # ====================================================
            # READ OLD DATASET
            # ====================================================

            try:

                old_df = pd.read_csv(
                    old_dataset
                )

            except Exception as exc:

                raise RuntimeError(
                    "Could not read old dataset "
                    "as CSV: "
                    + str(exc)
                )

            # ====================================================
            # READ NEW DATASET
            # ====================================================

            try:

                new_df = pd.read_csv(
                    new_dataset
                )

            except Exception as exc:

                raise RuntimeError(
                    "Could not read new dataset "
                    "as CSV: "
                    + str(exc)
                )

            # ====================================================
            # COLUMN COMPARISON
            # ====================================================

            old_columns = list(
                old_df.columns
            )

            new_columns = list(
                new_df.columns
            )

            columns_added = [
                column
                for column in new_columns
                if column not in old_columns
            ]

            columns_removed = [
                column
                for column in old_columns
                if column not in new_columns
            ]

            # ====================================================
            # COMMON COLUMNS
            # ====================================================

            common_columns = [
                column
                for column in old_columns
                if column in new_columns
            ]

            # ====================================================
            # ROW COMPARISON
            # ====================================================

            old_records = set(
                tuple(row)
                for row in old_df[
                    common_columns
                ].itertuples(
                    index=False,
                    name=None,
                )
            )

            new_records = set(
                tuple(row)
                for row in new_df[
                    common_columns
                ].itertuples(
                    index=False,
                    name=None,
                )
            )

            # ====================================================
            # ADDED ROWS
            # ====================================================

            added_records = (
                new_records - old_records
            )

            # ====================================================
            # REMOVED ROWS
            # ====================================================

            removed_records = (
                old_records - new_records
            )

            # ====================================================
            # CONVERT ROWS TO DICTIONARIES
            # ====================================================

            added_rows = [
                dict(
                    zip(
                        common_columns,
                        row,
                    )
                )
                for row in added_records
            ]

            removed_rows = [
                dict(
                    zip(
                        common_columns,
                        row,
                    )
                )
                for row in removed_records
            ]

            # ====================================================
            # RETURN COMPLETE RESULT
            # ====================================================

            return {
                "dataset":
                    actual_data_path,

                "dataset_changed":
                    True,

                "old_dvc_hash":
                    old_hash,

                "new_dvc_hash":
                    new_hash,

                "old_rows":
                    len(old_df),

                "new_rows":
                    len(new_df),

                "old_columns":
                    old_columns,

                "new_columns":
                    new_columns,

                "columns_added":
                    columns_added,

                "columns_removed":
                    columns_removed,

                "rows_added":
                    len(added_rows),

                "rows_removed":
                    len(removed_rows),

                "added_rows":
                    added_rows,

                "removed_rows":
                    removed_rows,
            }