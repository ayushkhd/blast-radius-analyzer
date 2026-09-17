"""Splits a cleaned document into retrieval chunks.

Most documents in this corpus are shorter than one chunk, so chunking is
mainly about the long tail: kernel CVE descriptions of up to 4,000
characters whose second half is often a call trace. The embedding model
reads at most 512 tokens, roughly 2,000 characters, so an unchunked long
description would have its tail silently ignored.

Chunks end at sentence boundaries where they can, carry a little overlap so
a fact split across a boundary is still found, and keep kernel traces in
chunks of their own (``kind == "trace"``) so that the caller can index them
for keyword search without embedding them.
"""

import dataclasses
from typing import Literal

from blast_radius import textproc

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_SENTENCES = 1

# New text shorter than this at the end of a document joins the chunk before
# it: a chunk that is nearly all overlap, or a lone trailing word, only adds
# a near-duplicate to the index.
_MIN_TAIL_CHARS = 100


@dataclasses.dataclass(frozen=True)
class Piece:
  """One chunk of a document.

  Attributes:
    kind: ``"text"`` for prose, ``"trace"`` for a kernel log excerpt.
    text: The chunk's content, whitespace-normalised.
  """

  kind: Literal["text", "trace"]
  text: str


def _split_hard(text: str, max_chars: int) -> list[str]:
  """Splits ``text`` on whitespace into parts of at most ``max_chars``.

  Used for text with no sentence structure: traces, and the rare sentence
  that is longer than a whole chunk. A single token longer than
  ``max_chars`` is kept whole rather than cut mid-word.

  Args:
    text: Whitespace-normalised text.
    max_chars: Upper bound on the length of each part.
  """
  parts: list[str] = []
  current = ""
  for word in text.split(" "):
    if current and len(current) + 1 + len(word) > max_chars:
      parts.append(current)
      current = word
    else:
      current = f"{current} {word}" if current else word
  if current:
    parts.append(current)
  return parts


def _pack_sentences(
    sentences: list[str], max_chars: int, overlap_sentences: int
) -> list[str]:
  """Packs sentences greedily into chunks of at most ``max_chars``.

  Each chunk after the first starts with the last ``overlap_sentences``
  sentences of the one before, unless that would leave no room for new
  text. A very short tail joins the previous chunk, which may then exceed
  ``max_chars`` by up to ``_MIN_TAIL_CHARS``.

  Args:
    sentences: Sentences in document order.
    max_chars: Upper bound on the length of each chunk.
    overlap_sentences: How many trailing sentences to repeat.
  """
  units: list[str] = []
  for sentence in sentences:
    if len(sentence) > max_chars:
      units.extend(_split_hard(sentence, max_chars))
    else:
      units.append(sentence)

  chunks: list[str] = []
  current: list[str] = []
  fresh = 0  # Units in ``current`` that are not overlap from the last chunk.
  for unit in units:
    if current and len(" ".join(current)) + 1 + len(unit) > max_chars:
      chunks.append(" ".join(current))
      overlap = current[-overlap_sentences:] if overlap_sentences else []
      if len(" ".join(overlap)) + 1 + len(unit) > max_chars:
        overlap = []
      current = list(overlap)
      fresh = 0
    current.append(unit)
    fresh += 1
  if fresh:
    tail = " ".join(current[-fresh:])
    if chunks and len(tail) < _MIN_TAIL_CHARS:
      chunks[-1] = f"{chunks[-1]} {tail}"
    else:
      chunks.append(" ".join(current))
  return chunks


def chunk_document(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[Piece]:
  """Returns the chunks of one cleaned document, in document order.

  Args:
    text: Plain text: HTML already stripped, boilerplate already removed.
    max_chars: Upper bound on the length of each chunk.
    overlap_sentences: Trailing sentences repeated at the start of the next
      prose chunk. Traces are never overlapped.

  Raises:
    ValueError: If ``max_chars`` is not positive or ``overlap_sentences`` is
      negative.
  """
  if max_chars <= 0:
    raise ValueError(f"max_chars must be positive, got {max_chars}")
  if overlap_sentences < 0:
    raise ValueError(
        f"overlap_sentences must not be negative, got {overlap_sentences}"
    )

  pieces: list[Piece] = []
  for is_trace, segment in textproc.segment_traces(text):
    if is_trace:
      flat = textproc.normalise_whitespace(segment)
      pieces.extend(
          Piece("trace", part) for part in _split_hard(flat, max_chars)
      )
    else:
      sentences = textproc.split_sentences(segment)
      pieces.extend(
          Piece("text", part)
          for part in _pack_sentences(sentences, max_chars, overlap_sentences)
      )
  return pieces
