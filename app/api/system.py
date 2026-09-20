import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import Project
from app.db.user_models import User


router = APIRouter(
    prefix="/system",
    tags=["System"],
)


WINDOWS_NO_WINDOW = (
    subprocess.CREATE_NO_WINDOW
    if os.name == "nt"
    else 0
)


# ------------------------------------------------------------
# Command helpers
# ------------------------------------------------------------


def _run_command(
    command: list[str],
    timeout: float = 8.0,
) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=WINDOWS_NO_WINDOW,
        )

    except (
        OSError,
        subprocess.SubprocessError,
    ):
        return None

    if result.returncode != 0:
        return None

    output = (
        result.stdout
        or result.stderr
    ).strip()

    return output or None


def _first_line(
    output: str | None,
) -> str | None:
    if not output:
        return None

    return (
        output
        .splitlines()[0]
        .strip()
        or None
    )


def _run_version_command(
    command: list[str],
) -> str | None:
    return _first_line(
        _run_command(command)
    )


def _current_python_path() -> str:
    return str(
        Path(
            sys.executable
        ).resolve()
    )


def _which_with_windows_fallbacks(
    binary: str,
) -> str | None:

    found = shutil.which(binary)

    if found:
        return str(
            Path(found).resolve()
        )

    # The DATAGIT backend and CLI are normally
    # installed in the same virtual environment.
    # On Windows, check that environment explicitly.
    if os.name == "nt":

        scripts_dir = (
            Path(
                sys.executable
            )
            .resolve()
            .parent
        )

        same_env_candidates = [
            scripts_dir
            / f"{binary}.exe",

            scripts_dir
            / f"{binary}.cmd",

            scripts_dir
            / binary,
        ]

        for candidate in same_env_candidates:

            if candidate.is_file():
                return str(
                    candidate.resolve()
                )

    # VS Code may not be on PATH.
    if (
        os.name != "nt"
        or binary != "code"
    ):
        return None

    candidates = [
        Path(
            os.environ.get(
                "LOCALAPPDATA",
                "",
            )
        )
        / "Programs"
        / "Microsoft VS Code"
        / "Code.exe",

        Path(
            os.environ.get(
                "PROGRAMFILES",
                "",
            )
        )
        / "Microsoft VS Code"
        / "Code.exe",

        Path(
            os.environ.get(
                "PROGRAMFILES(X86)",
                "",
            )
        )
        / "Microsoft VS Code"
        / "Code.exe",
    ]

    for candidate in candidates:

        if candidate.is_file():
            return str(
                candidate.resolve()
            )

    return None


# ------------------------------------------------------------
# Dependency detection
# ------------------------------------------------------------


def _dependency(
    *,
    dependency_id: str,
    name: str,
    description: str,
    required: bool,
    binary: str | None,
    version_command: list[str] | None,
    download_url: str,
) -> dict[str, Any]:

    path: str | None = None
    version: str | None = None

    installed = False

    if binary:

        path = (
            _which_with_windows_fallbacks(
                binary
            )
        )

        installed = (
            path is not None
        )

        if (
            installed
            and version_command
        ):
            version = (
                _run_version_command(
                    version_command
                )
            )

    return {
        "id": dependency_id,
        "name": name,
        "description": description,
        "required": required,
        "installed": installed,
        "version": version,
        "path": path,
        "download_url": download_url,
    }


def _python_version_text() -> str | None:

    output = _run_version_command(
        [
            _current_python_path(),
            "--version",
        ]
    )

    if output:
        return output

    return _run_version_command(
        [
            "python",
            "--version",
        ]
    )


def _python_meets_requirement(
    version_text: str | None,
) -> bool:

    if not version_text:
        return False

    try:
        raw = (
            version_text
            .replace(
                "Python ",
                "",
                1,
            )
            .strip()
        )

        parts = raw.split(".")

        major = int(parts[0])
        minor = int(parts[1])

    except (
        IndexError,
        TypeError,
        ValueError,
    ):
        return False

    return (
        major,
        minor,
    ) >= (
        3,
        12,
    )


def _detect_python_dependency(
) -> dict[str, Any]:

    path = _current_python_path()

    version = (
        _python_version_text()
    )

    return {
        "id": "python",
        "name": "Python 3.12+",
        "description": (
            "Python runtime used by "
            "DATAGIT and the user's "
            "ML projects."
        ),
        "required": True,
        "installed": (
            _python_meets_requirement(
                version
            )
        ),
        "version": version,
        "path": path,
        "download_url": (
            "https://www.python.org/downloads/"
        ),
    }


def _detect_python_extension(
    code_path: str | None,
) -> dict[str, Any]:

    result: dict[str, Any] = {

        "id":
            "vscode_python_extension",

        "name":
            "VS Code Python Extension",

        "description":
            (
                "Python language support "
                "and virtual-environment "
                "selection inside VS Code."
            ),

        "required":
            True,

        "installed":
            False,

        "version":
            None,

        "path":
            code_path,

        "download_url":
            (
                "https://marketplace."
                "visualstudio.com/"
                "items?itemName="
                "ms-python.python"
            ),
    }

    if not code_path:
        return result

    output = _run_command(
        [
            code_path,
            "--list-extensions",
            "--show-versions",
        ]
    )

    if not output:
        return result

    extension_line = next(
        (
            line.strip()
            for line in (
                output.splitlines()
            )
            if (
                line.strip()
                .lower()
                .startswith(
                    "ms-python.python"
                )
            )
        ),
        None,
    )

    if not extension_line:
        return result

    result["installed"] = True

    result["version"] = (
        extension_line.split(
            "@",
            1,
        )[1]
        if "@" in extension_line
        else "AVAILABLE"
    )

    return result


def _detect_keyring() -> dict[str, Any]:

    version: str | None = None

    try:

        version = (
            importlib.metadata.version(
                "keyring"
            )
        )

    except (
        importlib.metadata.PackageNotFoundError
    ):

        version = _first_line(
            _run_command(
                [
                    _current_python_path(),
                    "-c",
                    (
                        "import "
                        "importlib.metadata; "
                        "print("
                        "importlib.metadata."
                        "version('keyring')"
                        ")"
                    ),
                ]
            )
        )

    return {
        "id": "python_keyring",

        "name":
            "Python Keyring",

        "description":
            (
                "Secure local credential "
                "storage used for seamless "
                "DATAGIT CLI authentication."
            ),

        "required":
            True,

        "installed":
            version is not None,

        "version":
            version,

        "path":
            _current_python_path(),

        "download_url":
            (
                "https://pypi.org/"
                "project/keyring/"
            ),
    }


# ------------------------------------------------------------
# GET DEPENDENCY STATUS
# ------------------------------------------------------------


@router.get(
    "/dependencies"
)
def get_dependency_status(
    _current_user: User = Depends(
        get_current_user
    ),
):

    code_path = (
        _which_with_windows_fallbacks(
            "code"
        )
    )

    dependencies = [

        _detect_python_dependency(),

        _dependency(
            dependency_id="git",
            name="Git",
            description=(
                "Source-code history "
                "and Git provenance "
                "for DATAGIT versions."
            ),
            required=True,
            binary="git",
            version_command=[
                "git",
                "--version",
            ],
            download_url=(
                "https://git-scm.com/downloads"
            ),
        ),

        _dependency(
            dependency_id="dvc",
            name="DVC",
            description=(
                "Dataset versioning "
                "and immutable "
                "data-state references."
            ),
            required=True,
            binary="dvc",
            version_command=[
                "dvc",
                "--version",
            ],
            download_url=(
                "https://dvc.org/doc/install"
            ),
        ),

        _dependency(
            dependency_id="vscode",
            name="Visual Studio Code",
            description=(
                "Primary coding workspace "
                "for selected ML projects."
            ),
            required=True,
            binary="code",
            version_command=(
                [
                    code_path,
                    "--version",
                ]
                if code_path
                else None
            ),
            download_url=(
                "https://code.visualstudio.com/download"
            ),
        ),

        _detect_python_extension(
            code_path
        ),

        _dependency(
            dependency_id="datagit_cli",
            name="DATAGIT CLI",
            description=(
                "Command-line entry "
                "points for project "
                "and version operations."
            ),
            required=True,
            binary="enchanting",
            version_command=None,
            download_url=(
                "https://github.com/"
                "Soham-kumbhar/"
                "DATA-GIT-BACKEND"
            ),
        ),

        _detect_keyring(),
    ]

    return {
        "platform":
            platform.system(),

        "python_executable":
            _current_python_path(),

        "dependencies":
            dependencies,
    }


# ------------------------------------------------------------
# START BUILDING
# ------------------------------------------------------------


def _find_project_venv(
    project_path: Path,
) -> Path | None:

    candidates = (

        project_path
        / ".venv"
        / "Scripts"
        / "python.exe",

        project_path
        / "venv"
        / "Scripts"
        / "python.exe",
    )

    for candidate in candidates:

        if candidate.is_file():
            return candidate.resolve()

    return None


def _open_vscode(
    code_path: str,
    project_path: Path,
) -> None:

    creation_flags = (
        (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
        if os.name == "nt"
        else 0
    )

    subprocess.Popen(
        [
            code_path,
            "--reuse-window",
            str(project_path),
        ],
        cwd=str(project_path),
        close_fds=True,
        creationflags=creation_flags,
    )


@router.post(
    "/projects/{project_id}/start-building"
)
def start_building_project(
    project_id: int,

    db: Session = Depends(
        get_db
    ),

    current_user: User = Depends(
        get_current_user
    ),
):

    if (
        platform.system()
        != "Windows"
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "START BUILDING currently "
                "supports the Windows VS Code "
                "workflow."
            ),
        )

    project = (
        db.query(Project)
        .filter(
            Project.id
            == project_id,

            Project.user_id
            == current_user.id,
        )
        .first()
    )

    if project is None:

        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=
                "Project not found.",
        )

    project_path = (
        Path(
            project.path
        )
        .expanduser()
        .resolve()
    )

    if not project_path.is_dir():

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "The registered project "
                "folder no longer exists "
                "on this device."
            ),
        )

    code_path = (
        _which_with_windows_fallbacks(
            "code"
        )
    )

    if not code_path:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Visual Studio Code is not "
                "installed or the 'code' "
                "command is unavailable."
            ),
        )

    venv_python = (
        _find_project_venv(
            project_path
        )
    )

    try:

        _open_vscode(
            code_path,
            project_path,
        )

    except OSError as error:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Unable to open Visual "
                "Studio Code: "
                f"{error}"
            ),
        ) from error

    if venv_python:

        message = (
            "VS Code opened the selected "
            "project. Project virtual "
            "environment detected at "
            f"{venv_python.parent.parent}. "
            "Open a new Python terminal "
            "and VS Code can use this "
            "environment."
        )

    else:

        message = (
            "VS Code opened the selected "
            "project. No project virtual "
            "environment (.venv or venv) "
            "was detected."
        )

    return {
        "status":
            "started",

        "project_path":
            str(project_path),

        "vscode_path":
            code_path,

        "venv_detected":
            venv_python is not None,

        "venv_path":
            (
                str(
                    venv_python
                    .parent
                    .parent
                )
                if venv_python
                else None
            ),

        "message":
            message,
    }