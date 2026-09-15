import os

import click
import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"


def line(char="─", width=62):
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
        "┌" + line("─", width_left + width_right + 3) + "┐",
        fg="bright_black",
    )

    for key, value in rows:
        key_text = str(key).ljust(width_left)
        value_text = str(value)

        click.secho(
            f"│ {key_text} │ {value_text:<{width_right}} │",
            fg="white",
        )

    click.secho(
        "└" + line("─", width_left + width_right + 3) + "┘",
        fg="bright_black",
    )


def print_help_box(title, content_lines, color="cyan"):
    width = 62

    click.secho(
        "╭" + line("─", width) + "╮",
        fg=color,
    )

    title_text = f"  {title}"
    click.secho(
        "│" + title_text.ljust(width) + "│",
        fg=color,
        bold=True,
    )

    click.secho(
        "├" + line("─", width) + "┤",
        fg=color,
    )

    for content in content_lines:
        text = f"  {content}"
        click.secho(
            "│" + text.ljust(width) + "│",
            fg="white",
        )

    click.secho(
        "╰" + line("─", width) + "╯",
        fg=color,
    )


class DataGitGroup(click.Group):

    def format_help(self, ctx, formatter):
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


class VersionCommand(click.Command):

    def format_help(self, ctx, formatter):
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
                "      DataGit project ID.",
                "",
                "  --api-url URL",
                "      DataGit backend URL.",
                "",
                "WHAT HAPPENS",
                "  1. Capture current Git commit.",
                "  2. Capture current DVC state.",
                "  3. Create immutable Version.",
                "  4. Store your message.",
            ],
            color="bright_magenta",
        )

        click.echo()


@click.group(cls=DataGitGroup)
def cli():
    """DataGit — ML-aware Git and DVC versioning."""


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
    help="DataGit project ID.",
)
@click.option(
    "--api-url",
    default=DEFAULT_API_URL,
    show_default=False,
    metavar="URL",
    help="DataGit backend URL.",
)
def version(
    description: str,
    project_id: int | None,
    api_url: str,
):
    """Finalize the current Git + DVC state as a new Version."""

    if project_id is None:
        project_id_text = os.getenv("DATAGIT_PROJECT_ID")

        if not project_id_text:
            raise click.ClickException(
                "Project ID is required. "
                "Use --project-id or set DATAGIT_PROJECT_ID."
            )

        try:
            project_id = int(project_id_text)
        except ValueError as error:
            raise click.ClickException(
                "DATAGIT_PROJECT_ID must be an integer."
            ) from error

    description = description.strip()

    if not description:
        raise click.ClickException(
            "Version message cannot be empty."
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
            ("Project ID", project_id),
            ("Message", description),
            ("Backend", api_url),
        ]
    )

    click.echo()

    click.secho(
        "  Capturing Git + DVC state...",
        fg="cyan",
    )

    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/projects/"
            f"{project_id}/versions/finalize",
            json={
                "description": description,
            },
            timeout=30,
        )

    except requests.RequestException as error:
        click.echo()

        click.secho(
            "╭" + line("─", 62) + "╮",
            fg="bright_red",
        )

        click.secho(
            "│  ERROR                                                     │",
            fg="bright_red",
            bold=True,
        )

        click.secho(
            "├" + line("─", 62) + "┤",
            fg="bright_red",
        )

        click.secho(
            "│  Could not connect to DataGit.                            │",
            fg="white",
        )

        click.secho(
            "╰" + line("─", 62) + "╯",
            fg="bright_red",
        )

        raise click.ClickException(
            str(error)
        ) from error

    if response.status_code == 201:
        data = response.json()

        click.echo()

        click.secho(
            "╭" + line("─", 62) + "╮",
            fg="bright_green",
        )

        click.secho(
            "│  ✓ VERSION FINALIZED                                      │",
            fg="bright_green",
            bold=True,
        )

        click.secho(
            "├" + line("─", 62) + "┤",
            fg="bright_green",
        )

        print_table(
            [
                ("Version", data["version_number"]),
                ("Description", data["description"]),
                ("Git commit", data["git_commit"]),
            ]
        )

        click.secho(
            "╰" + line("─", 62) + "╯",
            fg="bright_green",
        )

        click.echo()

        click.secho(
            "  Your Git + DVC state is now an immutable DataGit Version.",
            fg="bright_cyan",
            bold=True,
        )

        click.echo()

        return

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
            "╭" + line("─", 62) + "╮",
            fg="bright_yellow",
        )

        click.secho(
            "│  • NOTHING TO FINALIZE                                    │",
            fg="bright_yellow",
            bold=True,
        )

        click.secho(
            "├" + line("─", 62) + "┤",
            fg="bright_yellow",
        )

        click.secho(
            f"│  {detail[:58]:<58}│",
            fg="white",
        )

        click.secho(
            "╰" + line("─", 62) + "╯",
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
        f"Finalize failed ({response.status_code}): {detail}"
    )


if __name__ == "__main__":
    cli()