"""Tests for the static UI in blast_radius/static.

The page shows third-party write-ups and model-written prose, and it is
served under a Content-Security-Policy that allows same-origin script and
style only. These tests pin that posture by reading the three files as
text. They run no browser.
"""

import html.parser
import pathlib
import re

import pytest

_STATIC = pathlib.Path(__file__).parent.parent / "blast_radius" / "static"
_FILES = ("index.html", "app.js", "styles.css")

# Ways of turning a string into markup, code or an inline style.
_SINKS = (
    r"innerHTML",
    r"outerHTML",
    r"insertAdjacentHTML",
    r"document\.write",
    r"DOMParser",
    r"\beval\s*\(",
    r"new\s+Function",
    r"cssText",
    r"setAttribute\(\s*[\"']style",
)


def _read(name: str) -> str:
  """Returns the text of one of the static files."""
  return (_STATIC / name).read_text(encoding="utf-8")


class _StartTags(html.parser.HTMLParser):
  """Collects every start tag of a document with its attributes."""

  def __init__(self) -> None:
    super().__init__()
    self.tags: list[tuple[str, dict[str, str | None]]] = []

  def handle_starttag(
      self, tag: str, attrs: list[tuple[str, str | None]]
  ) -> None:
    self.tags.append((tag, dict(attrs)))


def _start_tags(document: str) -> list[tuple[str, dict[str, str | None]]]:
  """Returns ``(tag, attributes)`` for every start tag in ``document``."""
  parser = _StartTags()
  parser.feed(document)
  return parser.tags


@pytest.mark.parametrize("sink", _SINKS)
def test_app_js_never_uses_a_markup_or_code_sink(sink: str):
  assert not re.search(sink, _read("app.js"))


def test_app_js_sets_a_link_target_in_one_place_with_a_safe_rel():
  script = _read("app.js")

  assert script.count(".href = ") == 1
  assert 'rel: "noopener noreferrer nofollow"' in script
  assert 'url.protocol === "http:" || url.protocol === "https:"' in script


def test_index_html_has_no_inline_script_style_or_event_handler():
  document = _read("index.html")

  tags = _start_tags(document)
  script_bodies = re.findall(r"<script\b[^>]*>(.*?)</script>", document, re.S)

  assert [body for body in script_bodies if body.strip()] == []
  assert "style" not in [tag for tag, _ in tags]
  for _, attributes in tags:
    assert "style" not in attributes
    assert not [name for name in attributes if name.startswith("on")]


def test_index_html_references_only_its_two_static_files():
  references = {
      value
      for _, attributes in _start_tags(_read("index.html"))
      for name, value in attributes.items()
      if name in ("src", "href") and value and not value.startswith("#")
  }

  assert references == {"/static/app.js", "/static/styles.css"}


@pytest.mark.parametrize("name", _FILES)
def test_static_file_names_no_absolute_url(name: str):
  assert not re.search(r"https?://", _read(name))


def test_styles_css_imports_nothing_and_loads_nothing_off_origin():
  styles = _read("styles.css")

  targets = re.findall(r"url\(\s*[\"']?([^\"')\s]+)", styles)

  assert "@import" not in styles
  for target in targets:
    is_rooted = target.startswith("/") and not target.startswith("//")
    assert is_rooted or target.startswith("data:")
