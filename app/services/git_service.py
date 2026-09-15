from pathlib import Path
import subprocess


class GitService:

    @staticmethod
    def _run_git(project_path: str, *args: str) -> str:
        """
        Run a Git command inside the given project.
        """
        result = subprocess.run(
            ["git", "-C", project_path, *args],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        return result.stdout.strip()

    @staticmethod
    def is_git_repository(project_path: str) -> bool:
        path = Path(project_path)

        if not path.exists() or not path.is_dir():
            return False

        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "rev-parse",
                "--is-inside-work-tree",
            ],
            capture_output=True,
            text=True,
        )

        return (
            result.returncode == 0
            and result.stdout.strip() == "true"
        )

    @staticmethod
    def get_current_branch(project_path: str) -> str:
        return GitService._run_git(
            project_path,
            "branch",
            "--show-current",
        )

    @staticmethod
    def get_current_commit(project_path: str) -> str:
        return GitService._run_git(
            project_path,
            "rev-parse",
            "HEAD",
        )

    @staticmethod
    def get_commits(
        project_path: str,
        limit: int = 20,
    ) -> list[dict]:

        output = GitService._run_git(
            project_path,
            "log",
            f"-{limit}",
            "--pretty=format:%H|%an|%ae|%ad|%s",
            "--date=iso",
        )

        if not output:
            return []

        commits = []

        for line in output.splitlines():
            commit_hash, author, email, date, message = line.split(
                "|",
                maxsplit=4,
            )

            commits.append(
                {
                    "hash": commit_hash,
                    "author": author,
                    "email": email,
                    "date": date,
                    "message": message,
                }
            )

        return commits

    @staticmethod
    def get_status(project_path: str) -> str:
        return GitService._run_git(
            project_path,
            "status",
            "--short",
        )

    @staticmethod
    def get_diff(project_path: str) -> str:
        return GitService._run_git(
            project_path,
            "diff",
        )

    @staticmethod
    def get_commit_diff(
        project_path: str,
        commit_hash: str,
    ) -> str:
        """
        Get the changes introduced by a specific commit.
        """
        return GitService._run_git(
            project_path,
            "show",
            "--format=",
            "--stat",
            "--patch",
            commit_hash,
        )

    @staticmethod
    def get_commit_patch(
        project_path: str,
        old_commit: str,
        new_commit: str,
    ) -> str:
        """
        Get the line-by-line changes between two commits.
        """
        return GitService._run_git(
            project_path,
            "diff",
            "--no-ext-diff",
            old_commit,
            new_commit,
        )

    @staticmethod
    def get_changed_files(
        project_path: str,
        old_commit: str,
        new_commit: str,
    ) -> list[str]:
        """
        Get file paths changed between two Git commits.
        """
        output = GitService._run_git(
            project_path,
            "diff",
            "--name-only",
            old_commit,
            new_commit,
        )

        if not output:
            return []

        return output.splitlines()