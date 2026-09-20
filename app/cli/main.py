import os

import click
import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"

KEYRING_SERVICE = "DATAGIT"

ACTIVE_USER_KEY = "active-user-id"

CLI_TOKEN_PREFIX = "cli-token:"


def line(
    char="─",
    width=62,
):
    return char * width


def print_header():
    click.echo()

    click.secho(
        "╭" + line("─", 62) + "╮",
        fg="bright_cyan",
    )

    click.secho(
        "│  DATAGIT                                                │",
        fg="bright_cyan",
        bold=True,
    )

    click.secho(
        "│  ML-aware Git + DVC versioning                          │",
        fg="cyan",
    )

    click.secho(
        "╰" + line("─", 62) + "╯",
        fg="bright_cyan",
    )

    click.echo()


def print_table(rows):
    width_left = 16
    width_right = 40

    click.secho(
        "┌" +
        line(
            "─",
            width_left + width_right + 3,
        ) +
        "┐",
        fg="bright_black",
    )

    for key, value in rows:
        key_text = str(key).ljust(
            width_left
        )

        value_text = str(value)

        click.secho(
            f"│ {key_text} │ "
            f"{value_text:<{width_right}} │",
            fg="white",
        )

    click.secho(
        "└" +
        line(
            "─",
            width_left + width_right + 3,
        ) +
        "┘",
        fg="bright_black",
    )


def print_help_box(
    title,
    content_lines,
    color="cyan",
):
    width = 62

    click.secho(
        "╭" +
        line(
            "─",
            width,
        ) +
        "╮",
        fg=color,
    )

    title_text = f"  {title}"

    click.secho(
        "│" +
        title_text.ljust(width) +
        "│",
        fg=color,
        bold=True,
    )

    click.secho(
        "├" +
        line(
            "─",
            width,
        ) +
        "┤",
        fg=color,
    )

    for content in content_lines:
        text = f"  {content}"

        click.secho(
            "│" +
            text.ljust(width) +
            "│",
            fg="white",
        )

    click.secho(
        "╰" +
        line(
            "─",
            width,
        ) +
        "╯",
        fg=color,
    )


def get_keyring_token() -> str | None:
    try:
        import keyring

        active_user_id = (
            keyring.get_password(
                KEYRING_SERVICE,
                ACTIVE_USER_KEY,
            )
        )

        if not active_user_id:
            return None

        try:
            user_id = int(
                active_user_id
            )
        except ValueError:
            return None

        return keyring.get_password(
            KEYRING_SERVICE,
            f"{CLI_TOKEN_PREFIX}{user_id}",
        )

    except Exception:
        return None


def clear_keyring_token():
    try:
        import keyring

        active_user_id = (
            keyring.get_password(
                KEYRING_SERVICE,
                ACTIVE_USER_KEY,
            )
        )

        if not active_user_id:
            return

        try:
            user_id = int(
                active_user_id
            )
        except ValueError:
            return

        try:
            keyring.delete_password(
                KEYRING_SERVICE,
                f"{CLI_TOKEN_PREFIX}{user_id}",
            )
        except Exception:
            pass

    except Exception:
        pass


def get_cli_token() -> str:
    """
    Authentication priority:

    1. Secure OS credential store.
    2. Legacy DATAGIT_ACCESS_TOKEN environment variable.

    New users do not need to set the environment variable.
    """

    token = get_keyring_token()

    if token:
        return token

    # Backward compatibility for existing installations.
    legacy_token = os.getenv(
        "DATAGIT_ACCESS_TOKEN"
    )

    if legacy_token:
        return legacy_token

    raise click.ClickException(
        "DATAGIT CLI is not connected to your signed-in account.\n\n"
        "Open DATAGIT and log in once. "
        "CLI access is provisioned automatically.\n"
        "You do not need to copy or paste a token."
    )


def request_headers(
    token: str,
) -> dict[str, str]:
    return {
        "Authorization": (
            f"Bearer {token}"
        ),
        "Content-Type": (
            "application/json"
        ),
    }


def fetch_projects(
    api_url: str,
    token: str,
):
    try:
        response = requests.get(
            f"{api_url.rstrip('/')}/projects",
            headers=request_headers(
                token
            ),
            timeout=30,
        )
    except requests.RequestException as error:
        raise click.ClickException(
            "Could not connect to DATAGIT backend.\n"
            f"{error}"
        ) from error

    if response.status_code == 401:
        clear_keyring_token()

        raise click.ClickException(
            "Your DATAGIT CLI session has expired.\n\n"
            "Open DATAGIT and log in again. "
            "CLI access will be provisioned automatically."
        )

    if response.status_code != 200:
        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        raise click.ClickException(
            f"Unable to load projects "
            f"({response.status_code}): {detail}"
        )

    try:
        return response.json()
    except ValueError as error:
        raise click.ClickException(
            "DATAGIT returned an invalid project response."
        ) from error


def resolve_project(
    api_url: str,
    token: str,
    project_number: int,
):
    projects = fetch_projects(
        api_url,
        token,
    )

    for project in projects:
        if (
            project.get(
                "project_number"
            )
            == project_number
        ):
            return project

    raise click.ClickException(
        f"DATAGIT Project {project_number} "
        "was not found for the currently signed-in user."
    )


class DataGitGroup(
    click.Group
):
    def format_help(
        self,
        ctx,
        formatter,
    ):
        print_help_box(
            "DATAGIT",
            [
                "ML-aware Git + DVC versioning for ML projects.",
                "",
                "COMMANDS",
                "  version     Finalize the current Git + DVC state",
                "",
                "VERSION COMMAND",
                '  enchanting version -m "MESSAGE"',
                "",
                "EXAMPLE",
                '  enchanting version -m "Added training samples"',
            ],
            color="bright_cyan",
        )

        click.echo()


class VersionCommand(
    click.Command
):
    def format_help(
        self,
        ctx,
        formatter,
    ):
        print_help_box(
            "VERSION",
            [
                "Finalize the current Git + DVC state.",
                "",
                "SYNTAX",
                '  enchanting version -m "MESSAGE"',
                "",
                "REQUIRED",
                "  -m, --message MESSAGE",
                "      Description stored with the Version.",
                "",
                "OPTIONAL",
                "  --project-id ID",
                "      Your user-facing DATAGIT project number.",
                "",
                "  --api-url URL",
                "      DATAGIT backend URL.",
                "",
                "WHAT HAPPENS",
                "  1. Authenticate current user.",
                "  2. Resolve your project number.",
                "  3. Capture current Git commit.",
                "  4. Capture current DVC state.",
                "  5. Create immutable Version.",
            ],
            color="bright_magenta",
        )

        click.echo()


@click.group(
    cls=DataGitGroup
)
def cli():
    """DATAGIT — ML-aware Git and DVC versioning."""


@cli.command(
    "version",
    cls=VersionCommand,
)
@click.option(
    "-m",
    "--message",
    "description",
    required=True,
    metavar="MESSAGE",
    help="Describe what changed in this version.",
)
@click.option(
    "--project-id",
    type=int,
    default=None,
    metavar="ID",
    help="Your user-facing DATAGIT project number.",
)
@click.option(
    "--api-url",
    default=DEFAULT_API_URL,
    show_default=False,
    metavar="URL",
    help="DATAGIT backend URL.",
)
def version(
    description: str,
    project_id: int | None,
    api_url: str,
):
    """
    Finalize the current Git + DVC state
    as a new DATAGIT Version.
    """

    if project_id is None:
        project_id_text = os.getenv(
            "DATAGIT_PROJECT_ID"
        )

        if not project_id_text:
            raise click.ClickException(
                "Project ID is required. "
                "Use --project-id or set DATAGIT_PROJECT_ID."
            )

        try:
            project_id = int(
                project_id_text
            )
        except ValueError as error:
            raise click.ClickException(
                "DATAGIT_PROJECT_ID must be an integer."
            ) from error

    description = description.strip()

    if not description:
        raise click.ClickException(
            "Version message cannot be empty."
        )

    token = get_cli_token()

    project = resolve_project(
        api_url=api_url,
        token=token,
        project_number=project_id,
    )

    internal_project_id = project.get(
        "id"
    )

    if not isinstance(
        internal_project_id,
        int,
    ):
        raise click.ClickException(
            "DATAGIT returned an invalid internal project identifier."
        )

    print_header()

    click.secho(
        "  FINALIZE VERSION",
        fg="bright_white",
        bold=True,
    )

    click.echo()

    print_table(
        [
            (
                "Project ID",
                project_id,
            ),
            (
                "Project",
                project.get(
                    "name",
                    "—",
                ),
            ),
            (
                "Message",
                description,
            ),
            (
                "Backend",
                api_url,
            ),
        ]
    )

    click.echo()

    click.secho(
        "  Capturing Git + DVC state...",
        fg="cyan",
    )

    try:
        response = requests.post(
            (
                f"{api_url.rstrip('/')}"
                f"/projects/"
                f"{internal_project_id}"
                f"/versions/finalize"
            ),
            headers=request_headers(
                token
            ),
            json={
                "description": description,
            },
            timeout=30,
        )

    except requests.RequestException as error:
        raise click.ClickException(
            "Could not connect to DATAGIT backend.\n"
            f"{error}"
        ) from error

    if response.status_code == 201:
        try:
            data = response.json()
        except ValueError as error:
            raise click.ClickException(
                "DATAGIT returned an invalid finalize response."
            ) from error

        click.echo()

        click.secho(
            "╭" +
            line(
                "─",
                62,
            ) +
            "╮",
            fg="bright_green",
        )

        click.secho(
            "│  ✓ VERSION FINALIZED                                      │",
            fg="bright_green",
            bold=True,
        )

        click.secho(
            "├" +
            line(
                "─",
                62,
            ) +
            "┤",
            fg="bright_green",
        )

        print_table(
            [
                (
                    "Project ID",
                    project_id,
                ),
                (
                    "Version",
                    data.get(
                        "version_number",
                        "—",
                    ),
                ),
                (
                    "Description",
                    data.get(
                        "description",
                        description,
                    ),
                ),
                (
                    "Git commit",
                    data.get(
                        "git_commit",
                        "—",
                    ),
                ),
            ]
        )

        click.secho(
            "╰" +
            line(
                "─",
                62,
            ) +
            "╯",
            fg="bright_green",
        )

        click.echo()

        click.secho(
            "  Your Git + DVC state is now "
            "an immutable DATAGIT Version.",
            fg="bright_cyan",
            bold=True,
        )

        click.echo()

        return

    if response.status_code == 401:
        clear_keyring_token()

        raise click.ClickException(
            "Your DATAGIT CLI session is no longer valid.\n\n"
            "Open DATAGIT and log in again. "
            "CLI access will be provisioned automatically."
        )

    if response.status_code == 403:
        raise click.ClickException(
            "You are not authorized to finalize this project."
        )

    if response.status_code == 404:
        try:
            detail = response.json().get(
                "detail",
                "Project or version endpoint not found.",
            )
        except ValueError:
            detail = response.text

        raise click.ClickException(
            str(detail)
        )

    if response.status_code == 409:
        try:
            detail = response.json().get(
                "detail",
                "This Git + DVC state is already finalized.",
            )
        except ValueError:
            detail = response.text

        click.echo()

        click.secho(
            "╭" +
            line(
                "─",
                62,
            ) +
            "╮",
            fg="bright_yellow",
        )

        click.secho(
            "│  • NOTHING TO FINALIZE                                    │",
            fg="bright_yellow",
            bold=True,
        )

        click.secho(
            "├" +
            line(
                "─",
                62,
            ) +
            "┤",
            fg="bright_yellow",
        )

        click.secho(
            f"│  {str(detail)[:58]:<58}│",
            fg="white",
        )

        click.secho(
            "╰" +
            line(
                "─",
                62,
            ) +
            "╯",
            fg="bright_yellow",
        )

        click.echo()

        return

    try:
        detail = response.json().get(
            "detail",
            response.text,
        )
    except ValueError:
        detail = response.text

    raise click.ClickException(
        f"Finalize failed "
        f"({response.status_code}): {detail}"
    )


if __name__ == "__main__":
    cli()