import os
from pathlib import Path
from typing import Literal, overload

CHANGELOG_PATH = Path("CHANGELOG.md")
RELEASE_NOTES_PATH = Path("release_notes.md")
UNRELEASED_HEADER = "## Unreleased"
SECTION_HEADERS = [
    "### Breaking changes",
    "### Enhancements",
    "### Bug fixes",
    "### Documentation",
    "### Miscellaneous",
]


@overload
def get_env_var(name: str, required: Literal[True]) -> str: ...
@overload
def get_env_var(name: str, required: Literal[False]) -> str | None: ...
def get_env_var(name: str, required: bool = False) -> str | None:
    """Fetches an environment variable.

    Args:
        name (str): The name of the environment variable.
        required (bool): If True, raises an error if the variable is not set.

    Raises:
        ValueError: If the variable is required but not set.

    Returns:
        str | None: The value of the environment variable, or None if not set.
    """
    value = os.getenv(name)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def parse_unreleased(changelog_lines: list[str]) -> tuple[list[str], int, int]:
    """Extracts the unreleased section and returns its lines, start index, and end index.

    Args:
        changelog_lines (list[str]): The lines of the changelog file.

    Raises:
        ValueError: If the '## Unreleased' section is not found.

    Returns:
        tuple[list[str], int, int]: A tuple containing the lines of the unreleased section,
                                     the start index, and the end index.
    """
    start = None
    end = None
    for i, line in enumerate(changelog_lines):
        if line.strip() == UNRELEASED_HEADER:
            start = i
        elif start is not None and line.startswith("## "):
            end = i
            break
    if start is None:
        raise ValueError("No '## Unreleased' section found in CHANGELOG.md")
    if end is None:
        end = len(changelog_lines)
    return changelog_lines[start:end], start, end


def extract_non_empty_sections(unreleased_lines: list[str]) -> list[str]:
    """Extracts non-empty sections from the unreleased lines.

    Args:
        unreleased_lines (list[str]): The lines of the unreleased section.

    Returns:
        list[str]: A list of non-empty sections.
    """
    current_section = None
    buffer = []
    result = []

    for line in unreleased_lines[1:]:  # Skip "## Unreleased"
        if any(line.startswith(h) for h in SECTION_HEADERS):
            if current_section and buffer:
                result.extend([current_section] + buffer + [""])
            current_section = line
            buffer = []
        elif line.strip() == "- None":
            continue
        elif line.strip().startswith("- "):
            buffer.append(line)

    if current_section and buffer:
        result.extend([current_section] + buffer + [""])

    return result


def construct_release_notes(description: str | None, sections: list[str]) -> list[str]:
    """Constructs release notes from the description and sections.

    Args:
        description (str | None): The release description.
        sections (list[str]): The sections to include in the release notes.

    Returns:
        list[str]: A list of formatted release notes.
    """
    notes = []
    if description:
        notes.append(description)
        notes.append("")
    notes.extend(sections)
    return notes


def update_changelog(
    changelog_lines: list[str], start: int, end: int, new_version: str, notes: list[str]
) -> list[str]:
    """Updates the changelog with the new version and notes.

    Args:
        changelog_lines (list[str]): The lines of the changelog file.
        start (int): The start index of the unreleased section.
        end (int): The end index of the unreleased section.
        new_version (str): The new version to be added.
        notes (list[str]): The release notes to be added.

    Returns:
        list[str]: The updated changelog lines.
    """
    updated = changelog_lines[:start]
    updated.append(UNRELEASED_HEADER)
    updated.append("")
    updated.extend(sum(([header, "- None", ""] for header in SECTION_HEADERS), []))
    updated.append(f"## v{new_version}")
    updated.extend([""] + notes)
    updated.extend(changelog_lines[end:])
    return updated


def main():
    release_version = get_env_var("RELEASE_VERSION", required=True)
    release_description = get_env_var("RELEASE_DESCRIPTION", required=False)

    changelog_lines = CHANGELOG_PATH.read_text(encoding="utf-8").splitlines()
    unreleased_section, start, end = parse_unreleased(changelog_lines)
    non_empty_sections = extract_non_empty_sections(unreleased_section)
    release_notes = construct_release_notes(release_description, non_empty_sections)
    updated_changelog = update_changelog(
        changelog_lines, start, end, release_version, release_notes
    )
    CHANGELOG_PATH.write_text("\n".join(updated_changelog) + "\n", encoding="utf-8")
    RELEASE_NOTES_PATH.write_text("\n".join(release_notes), encoding="utf-8")


if __name__ == "__main__":
    main()
