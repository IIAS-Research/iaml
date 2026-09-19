"""Check an installed distribution from outside its source checkout."""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import sysconfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_version", help="Expected PyIAML distribution version.")
    args = parser.parse_args()

    if not sys.flags.isolated:
        print("Installation check requires 'python -I'.", file=sys.stderr)
        return 1

    try:
        installed_version = version("PyIAML")
        if installed_version != args.expected_version:
            raise ValueError(
                f"Installed PyIAML version is {installed_version!r}; "
                f"expected {args.expected_version!r}."
            )

        import iaml
        from iaml import Dataset, IAML

        module_path = Path(iaml.__file__).resolve()
        prefix = Path(sys.prefix).resolve()
        install_paths = sysconfig.get_paths()
        site_packages = {
            Path(install_paths[name]).resolve() for name in ("purelib", "platlib")
        }
        if not module_path.is_relative_to(prefix) or not any(
            module_path.is_relative_to(path) for path in site_packages
        ):
            raise ValueError(
                f"iaml was imported from {module_path}, outside this "
                f"environment's site-packages ({prefix})."
            )

        IAML(max_workers=1)
    except (ImportError, PackageNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Installation check failed: {exc}", file=sys.stderr)
        return 1

    print(f"Installed PyIAML {installed_version}: {module_path}")
    print(f"{IAML.__name__} constructed; {Dataset.__name__} imported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
