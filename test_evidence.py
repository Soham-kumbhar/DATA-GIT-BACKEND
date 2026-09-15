from pprint import pprint

from app.services.evidence_service import EvidenceService


project = r"C:\Users\Admin\Desktop\my-ml-project"

commit = (
    "474e58dfeb8efb820bacaa1844e6b74b7a39f02c"
)

evidence = EvidenceService.get_project_evidence(
    project,
    commit,
)

pprint(evidence)