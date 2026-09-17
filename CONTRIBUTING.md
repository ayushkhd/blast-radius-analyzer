# Contributing

The code follows the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
as it is practised inside Google: 2-space indentation and 80 columns. The style
is enforced by tools, so the fastest way to get it right is to run them.

```bash
make install   # uv sync --all-extras
make fmt       # isort + pyink rewrite files to the house style
make check     # isort, pyink, pylint, mypy, pytest: everything CI runs
```

A change is ready when `make check` passes.

## Python

*   **Imports.** Import modules, not names: `from blast_radius import models`
    and then `models.Host`. The exceptions are `typing` and `collections.abc`,
    whose names may be imported directly. One import per line, sorted by isort.
    No relative imports.
*   **Types.** Every function is fully annotated, tests included where a
    parameter's type is not obvious from a fixture. mypy runs in strict mode.
    Prefer `X | None` to `Optional[X]` and built-in generics to `typing.List`.
*   **Docstrings.** Every module, class and public function has one. Function
    docstrings open with a one-line summary in the descriptive mood ("Returns
    the ...", "Splits ..."), then `Args:`, `Returns:` and `Raises:` as needed.
    pylint's docparams extension fails the build when an argument or a raised
    exception is missing. A `Returns:` section may be dropped when the summary
    line already starts with "Returns". Test functions take no docstring; their
    name says what they check.
*   **Comments** explain why, not what. If a comment restates the code, delete
    it.
*   **Naming.** `module_name`, `ClassName`, `function_name`, `CONSTANT_NAME`,
    with a leading underscore for anything private to its module. Exceptions
    end in `Error`.
*   **Strings.** Double quotes. f-strings for formatting, except in logging
    calls, which pass arguments lazily: `logging.info("built %d chunks", n)`.
*   **Errors.** Raise the most specific built-in or a module-level exception
    class. Never use a bare `except:`; catch the narrowest type that the code
    can actually handle. Validate at the edges and trust the types inside.
*   **State.** No mutable module-level state. Models in `blast_radius.models`
    are frozen; build a new one with `model_copy(update=...)`.
*   **SQL** is always parameterised. User text never reaches an FTS5 `MATCH`
    expression unescaped; use `blast_radius.retrieval.keyword`.
*   **Spelling** in prose and identifiers is British ("normalise"), to match
    the existing code.

`blast_radius/textproc.py`, `blast_radius/chunking.py` and their tests are the
reference for what finished code looks like here.

## Tests

*   pytest, one `tests/test_<module>.py` per module, named
    `test_<unit>_<condition>_<expectation>`.
*   Arrange, act, assert, separated by blank lines. One behaviour per test.
*   No test touches the network or downloads a model. Use
    `embeddings.HashingEmbedder`, `rerank.LexicalReranker` and the scripted
    provider in `tests/fakes.py`.
*   Tests run on the synthetic exports in `tests/fixtures/`, never on the
    reference dataset.

## Commits

One logical change per commit, and every commit passes `make check`.

```
area: summarise the change in the imperative, under 72 characters

Say what changed and, above all, why, in full sentences wrapped at 72
columns. Describe the behaviour a reader would otherwise have to work out
from the diff, the alternatives that were rejected, and anything that was
measured. Do not narrate the editing history.
```

`area` is the package or concern the commit touches: `build`, `ingest`,
`retrieval`, `pipeline`, `llm`, `api`, `ui`, `eval`, `docs`.
