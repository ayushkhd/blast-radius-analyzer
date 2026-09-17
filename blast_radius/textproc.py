"""Text cleaning for scanner diagnoses and CVE descriptions.

The corpus has three quirks that hurt retrieval if left alone:

1. Qualys diagnoses are HTML fragments (``<P>``, ``<BR>``, entities).
2. Most CVE descriptions are Linux kernel commit messages that open with
   the same sentence. Identical text pulls their embeddings together, so
   it is removed before chunking.
3. Many of those commit messages embed a call trace or register dump.
   That text is useful to keyword search (a function name is an exact
   term) and is noise to an embedding model, so ``segment_traces`` lets
   the chunker keep prose and trace apart.

Every function here is pure and deterministic.
"""

import html
import re

# A tag starts with a letter straight after "<" or "</". Requiring that keeps
# comparisons in prose and code ("a < b and c > d") from being read as tags.
_BLOCK_TAG = re.compile(
    r"</?(?:p|br|li|ul|ol|div|tr|h[1-6])(?![A-Za-z0-9])[^<>]*>", re.I
)
_ANY_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
_SPACES = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")
_WHITESPACE = re.compile(r"\s+")

_KERNEL_BOILERPLATE = re.compile(
    r"^\s*In the Linux kernel, the following vulnerability has been"
    r" resolved:\s*",
    re.I,
)

# A sentence ends at terminal punctuation followed by whitespace and
# something that looks like the start of a new sentence. Requiring the
# capital, digit or opening quote keeps "e.g. foo" and "v2.4. x" together
# often enough for chunking, where a bad split costs little.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9\"'(\[`])")

# Tokens that only appear in kernel logs: register dumps, symbol offsets,
# timestamps, hex words, opcode bytes, source locations and the markers that
# frame a trace. A two-digit opcode byte must contain a digit, so that "be"
# and "ad" in ordinary prose are not mistaken for one.
_TRACE_TOKEN = re.compile(
    r"""(?x)^(?:
        [A-Z][A-Z0-9_]{1,9}:                  # RIP: RAX: EFLAGS: ORIG_RAX:
      | (?:Code|Comm|Tainted|Workqueue):      # mixed-case log labels
      | (?:[0-9a-f]{4}:)?(?:0x)?[0-9a-f]{8,}[,;:.)]*   # 0018:ffffc9000927f0d8
      | \S+\+0x[0-9a-f]+/0x[0-9a-f]+\S*       # nfs_net_init+0x1a/0x2b0
      | \S+\.[chS]:\d+\S*                     # net/ipv6/ip6_output.c:358
      | \[\s*\d+\.\d+\]                       # [   12.345678]
      | \d+\.\d+\]                            # second half of "[ 12.3]"
      | \[<?[0-9a-z_]+>?\]                    # [nf_tables] [inline] [<ffff>]
      | </?(?:TASK|IRQ|NMI|SOFTIRQ)>          # <TASK>
      | <?(?=[0-9a-f]?\d)[0-9a-f]{2}>?        # 8b <0f> 1f: opcode bytes
      | \?                                    # "? symbol+0x..." prefix
    )$""",
    re.I,
)
_TRACE_WINDOW = 6
_TRACE_DENSITY = 0.4
_MIN_TRACE_TOKENS = 12
# Prose this short between two traces is log text the patterns missed
# ("Allocated by task 506:"), not a sentence worth embedding.
_MIN_PROSE_BETWEEN_TRACES = 120


def normalise_whitespace(text: str) -> str:
  """Returns ``text`` with runs of whitespace collapsed to one space."""
  return _WHITESPACE.sub(" ", text).strip()


def strip_html(text: str) -> str:
  """Returns ``text`` as plain text, with paragraphs kept as blank lines.

  Block-level tags become paragraph breaks, every other tag is dropped, and
  entities are decoded. Input that is already plain text passes through
  with only its whitespace tidied.

  This is for Qualys diagnoses, which are HTML fragments. NVD descriptions
  are plain text in which ``<TASK>`` or ``<linux/foo.h>`` is content, not
  markup, so they must not be passed through here.

  Args:
    text: An HTML fragment or plain text.
  """
  text = _BLOCK_TAG.sub("\n\n", text)
  text = _ANY_TAG.sub(" ", text)
  text = html.unescape(text).replace("\xa0", " ")
  lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
  return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def remove_kernel_boilerplate(text: str) -> str:
  """Returns ``text`` without the sentence that opens kernel CVEs."""
  return _KERNEL_BOILERPLATE.sub("", text, count=1)


def split_sentences(text: str) -> list[str]:
  """Returns the sentences of ``text`` in order, none of them empty.

  Paragraph breaks always end a sentence. The splitter is a heuristic: it
  is used to find places where a chunk may end, so an occasional missed or
  extra boundary changes where a chunk is cut and nothing else.

  Args:
    text: Plain text, possibly with blank-line paragraph breaks.
  """
  sentences: list[str] = []
  for paragraph in _BLANK_LINES.split(text):
    for sentence in _SENTENCE_BOUNDARY.split(paragraph):
      sentence = normalise_whitespace(sentence)
      if sentence:
        sentences.append(sentence)
  return sentences


def _is_trace_token(token: str) -> bool:
  """Returns whether ``token`` looks like part of a kernel log."""
  return _TRACE_TOKEN.match(token) is not None


def segment_traces(text: str) -> list[tuple[bool, str]]:
  """Splits ``text`` into alternating prose and kernel-trace segments.

  A token belongs to a trace when trace-like tokens make up at least
  ``_TRACE_DENSITY`` of the window around it, which tolerates the symbol
  names and words that sit between the hex. Trace runs shorter than
  ``_MIN_TRACE_TOKENS`` are folded back into prose, so one stray address
  in a sentence does not split it, and a short prose fragment between two
  traces is folded into them.

  Paragraph structure is preserved inside prose segments so that sentence
  splitting still sees it.

  Args:
    text: Plain text.

  Returns:
    ``(is_trace, segment)`` pairs in document order, with no empty
    segments. Text without a trace yields a single prose segment.
  """
  # Tokens keep their leading whitespace so that prose is reassembled
  # exactly, paragraph breaks included.
  pieces = re.findall(r"\s*\S+", text)
  if not pieces:
    return []
  flags = [_is_trace_token(piece.strip()) for piece in pieces]

  in_trace = []
  for i in range(len(pieces)):
    lo = max(0, i - _TRACE_WINDOW)
    hi = min(len(pieces), i + _TRACE_WINDOW + 1)
    in_trace.append(sum(flags[lo:hi]) / (hi - lo) >= _TRACE_DENSITY)

  runs: list[tuple[bool, int, int]] = []
  start = 0
  for i in range(1, len(pieces) + 1):
    if i == len(pieces) or in_trace[i] != in_trace[start]:
      runs.append((in_trace[start], start, i))
      start = i

  # The window blurs each boundary by a few tokens. Snap every trace run so
  # that it starts and ends on a token that is itself trace-like, fold runs
  # that are too short back into prose, then merge neighbours of one kind.
  kinds = [False] * len(pieces)
  for is_trace, lo, hi in runs:
    if not is_trace:
      continue
    while lo > 0 and flags[lo - 1]:
      lo -= 1
    while hi < len(pieces) and flags[hi]:
      hi += 1
    while lo < hi and not flags[lo]:
      lo += 1
    while hi > lo and not flags[hi - 1]:
      hi -= 1
    if hi - lo >= _MIN_TRACE_TOKENS:
      kinds[lo:hi] = [True] * (hi - lo)

  merged: list[tuple[bool, int, int]] = []
  for i, is_trace in enumerate(kinds):
    if merged and merged[-1][0] == is_trace:
      merged[-1] = (is_trace, merged[-1][1], i + 1)
    else:
      merged.append((is_trace, i, i + 1))

  segments: list[tuple[bool, str]] = []
  for is_trace, lo, hi in merged:
    segment = "".join(pieces[lo:hi]).strip()
    if segment:
      segments.append((is_trace, segment))
  return _absorb_log_fragments(segments)


def _absorb_log_fragments(
    segments: list[tuple[bool, str]],
) -> list[tuple[bool, str]]:
  """Folds short prose that sits between two traces into those traces.

  Args:
    segments: ``(is_trace, segment)`` pairs in document order.

  Returns:
    The segments with each absorbed fragment joined to its neighbours.
  """
  relabelled = []
  for i, (is_trace, segment) in enumerate(segments):
    between = 0 < i < len(segments) - 1
    if not is_trace and between and len(segment) < _MIN_PROSE_BETWEEN_TRACES:
      is_trace = True
    relabelled.append((is_trace, segment))

  result: list[tuple[bool, str]] = []
  for is_trace, segment in relabelled:
    if result and result[-1][0] == is_trace:
      result[-1] = (is_trace, f"{result[-1][1]} {segment}")
    else:
      result.append((is_trace, segment))
  return result
