from app.services.git_service import GitService


PROJECT_PATH = r"C:\Users\Admin\Desktop\my-ml-project"


print("Is Git repository:")
print(GitService.is_git_repository(PROJECT_PATH))

print("\nCurrent branch:")
print(GitService.get_current_branch(PROJECT_PATH))

print("\nCurrent commit:")
print(GitService.get_current_commit(PROJECT_PATH))

print("\nCommit history:")
for commit in GitService.get_commits(PROJECT_PATH):
    print(commit)

print("\nGit status:")
print(GitService.get_status(PROJECT_PATH))

print("\nGit diff:")
print(GitService.get_diff(PROJECT_PATH))