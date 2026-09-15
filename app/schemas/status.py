from pydantic import BaseModel


class ProjectStatusResponse(BaseModel):
    project_id: int
    latest_version: int | None
    git_changed: bool
    dvc_changed: bool
    changes: list[str]