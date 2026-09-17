"""Tests for the package metadata."""

import pathlib
import tomllib

import blast_radius

_PYPROJECT = pathlib.Path(__file__).parent.parent / "pyproject.toml"


def test_version_matches_pyproject():
  project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]

  assert blast_radius.__version__ == project["version"]
