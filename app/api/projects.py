import ctypes
from ctypes import wintypes

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.user_models import User
from app.schemas.project import (
    ClaimLegacyProjectRequest,
    ProjectCreate,
    ProjectResponse,
)
from app.services.project_service import (
    ProjectConflictError,
    ProjectService,
)


router = APIRouter(
    prefix="/projects",
    tags=["Projects"],
)


# ============================================================
# WINDOWS NATIVE FOLDER PICKER
# ============================================================

if hasattr(ctypes, "windll"):
    ole32 = ctypes.windll.ole32
    user32 = ctypes.windll.user32
    shcore = getattr(
        ctypes.windll,
        "shcore",
        None,
    )
else:
    ole32 = None
    user32 = None
    shcore = None


CLSID_FILE_OPEN_DIALOG = (
    "{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}"
)

IID_IFILE_DIALOG = (
    "{42F85136-DB7E-439C-85F1-E4075D135FC8}"
)

IID_ISHELL_ITEM = (
    "{43826D1E-E718-42EE-BC55-A1E261C37BFE}"
)

SIGDN_FILESYSPATH = 0x80058000

FOS_PICKFOLDERS = 0x00000020
FOS_FORCEFILESYSTEM = 0x00000040
FOS_PATHMUSTEXIST = 0x00000800


def _guid_from_string(
    value: str,
):
    class GUID(ctypes.Structure):
        _fields_ = [
            (
                "Data1",
                wintypes.DWORD,
            ),
            (
                "Data2",
                wintypes.WORD,
            ),
            (
                "Data3",
                wintypes.WORD,
            ),
            (
                "Data4",
                wintypes.BYTE * 8,
            ),
        ]

    if ole32 is None:
        raise RuntimeError(
            "Windows COM is unavailable."
        )

    guid = GUID()

    ole32.CLSIDFromString(
        wintypes.LPCWSTR(value),
        ctypes.byref(guid),
    )

    return guid


def _get_com_method(
    instance,
    index,
    restype,
    argtypes,
):
    vtable = ctypes.cast(
        instance,
        ctypes.POINTER(
            ctypes.POINTER(
                ctypes.c_void_p
            )
        ),
    ).contents

    address = vtable[index]

    return ctypes.WINFUNCTYPE(
        restype,
        ctypes.c_void_p,
        *argtypes,
    )(address)


def _release_com_object(
    instance,
):
    if instance:
        try:
            release = _get_com_method(
                instance,
                2,
                wintypes.ULONG,
                [],
            )

            release(instance)

        except Exception:
            pass


def _set_windows_dpi_awareness():
    """
    Ask Windows for Per-Monitor-V2 DPI awareness.

    Failure is intentionally ignored because Windows
    versions differ in available DPI APIs.
    """

    if user32 is not None:
        try:
            set_context = (
                user32.SetProcessDpiAwarenessContext
            )

            set_context.argtypes = [
                ctypes.c_void_p,
            ]

            set_context.restype = (
                wintypes.BOOL
            )

            set_context(
                ctypes.c_void_p(-4)
            )

        except Exception:
            pass

    if shcore is not None:
        try:
            shcore.SetProcessDpiAwareness(
                ctypes.c_int(2)
            )

        except Exception:
            pass


def _select_folder_windows():
    """
    Open the native Windows Explorer-style
    folder picker.

    Returns:
        Absolute folder path, or None when cancelled.
    """

    if ole32 is None:
        raise RuntimeError(
            "Windows COM is unavailable."
        )

    _set_windows_dpi_awareness()

    result = ole32.CoInitializeEx(
        None,
        0x2,
    )

    com_initialized = result in (
        0,
        1,
    )

    dialog = None
    shell_item = None
    path_ptr = None

    try:
        dialog = ctypes.c_void_p()

        clsid = _guid_from_string(
            CLSID_FILE_OPEN_DIALOG
        )

        iid = _guid_from_string(
            IID_IFILE_DIALOG
        )

        hr = ole32.CoCreateInstance(
            ctypes.byref(clsid),
            None,
            0x1,
            ctypes.byref(iid),
            ctypes.byref(dialog),
        )

        if hr != 0:
            raise RuntimeError(
                "Unable to create Windows folder dialog. "
                f"HRESULT: {hr}"
            )

        # IFileDialog::GetOptions
        get_options = _get_com_method(
            dialog,
            10,
            ctypes.c_long,
            [
                ctypes.POINTER(
                    wintypes.DWORD
                ),
            ],
        )

        # IFileDialog::SetOptions
        set_options = _get_com_method(
            dialog,
            9,
            ctypes.c_long,
            [
                wintypes.DWORD,
            ],
        )

        options = wintypes.DWORD()

        hr = get_options(
            dialog,
            ctypes.byref(options),
        )

        if hr != 0:
            raise RuntimeError(
                "Unable to read Windows dialog options. "
                f"HRESULT: {hr}"
            )

        new_options = (
            options.value
            | FOS_PICKFOLDERS
            | FOS_FORCEFILESYSTEM
            | FOS_PATHMUSTEXIST
        )

        hr = set_options(
            dialog,
            new_options,
        )

        if hr != 0:
            raise RuntimeError(
                "Unable to configure Windows folder dialog. "
                f"HRESULT: {hr}"
            )

        # IFileDialog::SetTitle
        set_title = _get_com_method(
            dialog,
            17,
            ctypes.c_long,
            [
                wintypes.LPCWSTR,
            ],
        )

        hr = set_title(
            dialog,
            "Select DATAGIT Project Folder",
        )

        if hr != 0:
            raise RuntimeError(
                "Unable to set Windows folder dialog title. "
                f"HRESULT: {hr}"
            )

        # IModalWindow::Show
        show = _get_com_method(
            dialog,
            3,
            ctypes.c_long,
            [
                wintypes.HWND,
            ],
        )

        hr = show(
            dialog,
            None,
        )

        if hr == 0x800704C7:
            return None

        if hr != 0:
            raise RuntimeError(
                "Windows folder dialog failed. "
                f"HRESULT: {hr}"
            )

        # IFileDialog::GetResult
        get_result = _get_com_method(
            dialog,
            20,
            ctypes.c_long,
            [
                ctypes.POINTER(
                    ctypes.c_void_p
                ),
            ],
        )

        shell_item = ctypes.c_void_p()

        hr = get_result(
            dialog,
            ctypes.byref(shell_item),
        )

        if hr != 0:
            raise RuntimeError(
                "Unable to retrieve selected folder. "
                f"HRESULT: {hr}"
            )

        # IShellItem::GetDisplayName
        get_display_name = _get_com_method(
            shell_item,
            5,
            ctypes.c_long,
            [
                wintypes.DWORD,
                ctypes.POINTER(
                    wintypes.LPWSTR
                ),
            ],
        )

        path_ptr = wintypes.LPWSTR()

        hr = get_display_name(
            shell_item,
            SIGDN_FILESYSPATH,
            ctypes.byref(path_ptr),
        )

        if hr != 0:
            raise RuntimeError(
                "Unable to read selected folder path. "
                f"HRESULT: {hr}"
            )

        if not path_ptr.value:
            raise RuntimeError(
                "Windows returned an empty folder path."
            )

        return path_ptr.value

    finally:

        if path_ptr:
            try:
                ole32.CoTaskMemFree(
                    path_ptr
                )
            except Exception:
                pass

        _release_com_object(
            shell_item
        )

        _release_com_object(
            dialog
        )

        if com_initialized:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


# ============================================================
# PROJECT FOLDER SELECTION
# ============================================================

@router.get(
    "/select-folder",
)
def select_project_folder():
    try:
        selected_path = (
            _select_folder_windows()
        )

        return {
            "path": selected_path,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to open the Windows folder selector: "
                f"{error}"
            ),
        ) from error


# ============================================================
# CREATE PROJECT
# ============================================================

@router.post(
    "",
    response_model=ProjectResponse,
    status_code=201,
)
def create_project(
    project_data: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    try:
        return ProjectService.create_project(
            db=db,
            user_id=current_user.id,
            project_data=project_data,
        )

    except ProjectConflictError as error:
        detail = {
            "code": error.code,
            "message": error.message,
        }

        if error.project_id is not None:
            detail["project_id"] = (
                error.project_id
            )

        raise HTTPException(
            status_code=409,
            detail=detail,
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


# ============================================================
# CLAIM LEGACY PROJECT
# ============================================================

@router.post(
    "/claim-legacy",
    response_model=ProjectResponse,
)
def claim_legacy_project(
    request: ClaimLegacyProjectRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    try:
        return ProjectService.claim_legacy_project(
            db=db,
            user_id=current_user.id,
            project_path=request.path,
        )

    except ProjectConflictError as error:
        detail = {
            "code": error.code,
            "message": error.message,
        }

        if error.project_id is not None:
            detail["project_id"] = (
                error.project_id
            )

        raise HTTPException(
            status_code=409,
            detail=detail,
        ) from error


# ============================================================
# LIST PROJECTS
# ============================================================

@router.get(
    "",
    response_model=list[ProjectResponse],
)
def get_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    return ProjectService.get_projects(
        db=db,
        user_id=current_user.id,
    )


# ============================================================
# GET ONE PROJECT
# ============================================================

@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
)
def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    project = ProjectService.get_project(
        db=db,
        user_id=current_user.id,
        project_id=project_id,
    )

    if project is None:
        raise HTTPException(
            status_code=404,
            detail="Project not found.",
        )

    return project


# ============================================================
# DELETE PROJECT
# ============================================================

@router.delete(
    "/{project_id}",
    status_code=204,
)
def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    deleted = ProjectService.delete_project(
        db=db,
        user_id=current_user.id,
        project_id=project_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Project not found.",
        )

    return None