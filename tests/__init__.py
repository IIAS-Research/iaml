"""Test package entry point."""

# Import the real package before discovery: test helpers must not hide broken
# production imports by substituting packages in sys.modules.
import iaml  # noqa: F401
