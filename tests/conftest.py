"""Fixtures shared by every test module.

The tests run on the synthetic exports in ``tests/fixtures``, which
``tests/fixtures/make_fixtures.py`` documents and generates. The index
artifact is built from them once per session, with the model-free
``HashingEmbedder``, so no test touches the network or downloads a model.
"""

from collections.abc import Iterator
import pathlib

import pytest

from blast_radius import chunking
from blast_radius import embeddings
from blast_radius import ingest
from blast_radius import store as store_lib

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(name="assets_path", scope="session")
def fixture_assets_path() -> pathlib.Path:
  """Returns the committed synthetic asset export."""
  return _FIXTURES / "assets.json"


@pytest.fixture(name="vulns_path", scope="session")
def fixture_vulns_path() -> pathlib.Path:
  """Returns the committed synthetic vulnerability export."""
  return _FIXTURES / "vulns.json"


@pytest.fixture(name="artifact_path", scope="session")
def fixture_artifact_path(
    tmp_path_factory: pytest.TempPathFactory,
    assets_path: pathlib.Path,
    vulns_path: pathlib.Path,
) -> pathlib.Path:
  """Returns an artifact built from the synthetic exports, once per session.

  The artifact is shared, so tests must only read it. A test that needs a
  different build makes its own under ``tmp_path``.

  Args:
    tmp_path_factory: pytest's session-scoped temporary directories.
    assets_path: The synthetic asset export.
    vulns_path: The synthetic vulnerability export.
  """
  path = tmp_path_factory.mktemp("artifact") / "index.sqlite"
  ingest.build_artifact(
      assets_path,
      vulns_path,
      path,
      embeddings.HashingEmbedder(),
      chunk_max_chars=chunking.DEFAULT_MAX_CHARS,
      chunk_overlap_sentences=chunking.DEFAULT_OVERLAP_SENTENCES,
  )
  return path


@pytest.fixture(name="store")
def fixture_store(artifact_path: pathlib.Path) -> Iterator[store_lib.Store]:
  """Yields a store on the session's artifact, closed when the test ends."""
  with store_lib.Store(artifact_path) as opened:
    yield opened
