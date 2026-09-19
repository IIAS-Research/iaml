"""Check that release metadata agrees on the package version."""

import argparse
import json
from pathlib import Path
import sys
import tomllib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing the release metadata.",
    )
    parser.add_argument("--tag", help="Release tag to check, such as v1.0.0.")
    args = parser.parse_args()

    try:
        with (args.root / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)
        with (args.root / "uv.lock").open("rb") as stream:
            lock = tomllib.load(stream)
        manifest = json.loads(
            (args.root / ".release-please-manifest.json").read_text(encoding="utf-8")
        )

        version = project.get("project", {}).get("version")
        if not isinstance(version, str) or not version:
            raise ValueError("pyproject.toml must declare a nonempty project.version.")
        if not isinstance(manifest, dict) or manifest.get(".") != version:
            raise ValueError(
                ".release-please-manifest.json must set '.' to "
                f"the project version {version!r}."
            )

        packages = lock.get("package", [])
        if not isinstance(packages, list):
            raise ValueError("uv.lock must contain a list of package records.")
        matches = [
            package
            for package in packages
            if isinstance(package, dict) and package.get("name") == "pyiaml"
        ]
        if len(matches) != 1:
            raise ValueError(
                "uv.lock must contain exactly one 'pyiaml' package record; "
                f"found {len(matches)}."
            )
        locked_version = matches[0].get("version")
        if locked_version != version:
            raise ValueError(
                f"uv.lock has PyIAML version {locked_version!r}, "
                f"but pyproject.toml declares {version!r}."
            )
        if args.tag is not None and args.tag != f"v{version}":
            raise ValueError(
                f"Release tag {args.tag!r} must be 'v{version}' "
                "to match the package version."
            )
    except (OSError, ValueError) as exc:
        print(f"Version check failed: {exc}", file=sys.stderr)
        return 1

    print(f"Version check passed: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
