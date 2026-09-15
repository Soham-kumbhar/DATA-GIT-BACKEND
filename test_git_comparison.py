from app.services.git_service import GitService


project = r"C:\Users\Admin\Desktop\my-ml-project"

old_commit = "2462d8cd6f7b2ec2414d3748a3cf7bef52dda9ca"

new_commit = GitService.get_current_commit(project)

print("OLD COMMIT:")
print(old_commit)

print("\nNEW COMMIT:")
print(new_commit)

print("\nFILE CHANGES:")
print(
    GitService.get_commit_diff(
        project,
        old_commit,
        new_commit,
    )
)

print("\nACTUAL PATCH:")
print(
    GitService.get_commit_patch(
        project,
        old_commit,
        new_commit,
    )
)