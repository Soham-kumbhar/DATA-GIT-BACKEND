from pprint import pprint

from app.services.dvc_service import DVCService


project = r"C:\Users\Admin\Desktop\my-ml-project"

print("DVC state:")
pprint(
    DVCService.get_state(project)
)