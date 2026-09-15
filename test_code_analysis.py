from app.services.git_service import GitService
from app.services.code_analysis_service import CodeAnalysisService


project = r"C:\Users\Admin\Desktop\my-ml-project"

old_commit = "87c9d1366c92484409a106bba14e2c175f6d6307"
new_commit = "474e58dfeb8efb820bacaa1844e6b74b7a39f02c"

patch = GitService.get_commit_patch(
    project,
    old_commit,
    new_commit,
)

print("ACTUAL PATCH:")
print(patch)

print("\n" + "=" * 60)
print("CODE ANALYSIS:")
print(
    CodeAnalysisService.analyze_patch(patch)
)