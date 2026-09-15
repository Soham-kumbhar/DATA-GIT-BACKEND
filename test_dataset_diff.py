from app.services.dvc_service import DVCService


project = r"C:\Users\Admin\Desktop\my-ml-project"

old_commit = "2462d8cd6f7b2ec2414d3748a3cf7bef52dda9ca"
new_commit = "3e606ba2f57c148838cab94638b643328b7a0587"

result = DVCService.get_dataset_diff(
    project_path=project,
    old_commit=old_commit,
    new_commit=new_commit,
    data_path="data/cars.csv",
)

print("====================================================")
print("DATASET DIFF")
print("====================================================")

print(result)