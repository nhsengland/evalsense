import os
import sys
import tomllib

input_version = os.environ.get("INPUT_VERSION")
pyproject_file = "pyproject.toml"

try:
    with open(pyproject_file, "rb") as f:
        pyproject_data = tomllib.load(f)
except FileNotFoundError:
    print(f"::error::File '{pyproject_file}' not found.")
    sys.exit(1)
except tomllib.TOMLDecodeError:
    print(f"::error::Error decoding '{pyproject_file}'.")
    sys.exit(1)

try:
    project_version = pyproject_data["project"]["version"]
except KeyError:
    print("::error::Version not found in 'pyproject.toml'.")
    sys.exit(1)

if input_version != project_version:
    print(
        f"::error::Input version '{input_version}' does not match pyproject.toml version '{project_version}'."
    )
    sys.exit(1)

print(f"Version check passed: '{input_version}' matches '{project_version}'.")
