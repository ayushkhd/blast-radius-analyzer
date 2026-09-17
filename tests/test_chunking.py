"""Tests for blast_radius.chunking."""

import pytest

from blast_radius import chunking

_TRACE = (
    "RIP: 0010:ip6_output+0x231/0x3f0 net/ipv6/ip6_output.c:237"
    " Code: 3c 1e 00 49 89 df 74 08 4c 89 ef e8 23 4c 8b 74 24 28"
    " RSP: 0018:ffffc9000927f0d8 EFLAGS: 00010202"
    " RAX: 00000000000000bc RBX: 00000000000005e0 RCX: 0000000000040000"
    " Call Trace: <TASK> ip6_xmit+0xefe/0x17f0 net/ipv6/ip6_output.c:358"
    " sctp_v6_xmit+0x9f2/0x13f0 net/sctp/ipv6.c:248 </TASK>"
)


def _sentences(count: int, length: int = 100) -> list[str]:
  """Returns ``count`` distinct sentences of exactly ``length`` characters."""
  return [f"S{i:03d} ".ljust(length - 1, "x") + "." for i in range(count)]


def test_short_document_is_a_single_text_chunk():
  pieces = chunking.chunk_document("Traefik is a proxy. It can crash.")

  assert pieces == [chunking.Piece("text", "Traefik is a proxy. It can crash.")]


def test_empty_document_has_no_chunks():
  assert not chunking.chunk_document("  \n ")


def test_long_document_is_split_at_sentence_boundaries_within_the_limit():
  sentences = _sentences(10)

  pieces = chunking.chunk_document(
      " ".join(sentences), max_chars=350, overlap_sentences=0
  )

  assert [p.kind for p in pieces] == ["text"] * 4
  assert all(len(p.text) <= 350 for p in pieces)
  assert " ".join(p.text for p in pieces) == " ".join(sentences)


def test_adjacent_chunks_share_the_overlap_sentence():
  sentences = _sentences(6)

  pieces = chunking.chunk_document(
      " ".join(sentences), max_chars=350, overlap_sentences=1
  )

  assert pieces[0].text == " ".join(sentences[0:3])
  assert pieces[1].text == " ".join(sentences[2:5])
  assert pieces[1].text.startswith(sentences[2])


def test_overlap_is_dropped_when_it_would_leave_no_room_for_new_text():
  long_sentence = "A" + "a" * 298 + "."
  next_sentence = "B" + "b" * 198 + "."

  pieces = chunking.chunk_document(
      f"{long_sentence} {next_sentence}", max_chars=320, overlap_sentences=1
  )

  assert [p.text for p in pieces] == [long_sentence, next_sentence]


def test_short_tail_joins_the_previous_chunk_instead_of_standing_alone():
  sentences = _sentences(3) + ["Short tail."]

  pieces = chunking.chunk_document(
      " ".join(sentences), max_chars=310, overlap_sentences=0
  )

  assert len(pieces) == 1
  assert pieces[0].text.endswith("Short tail.")


def test_sentence_longer_than_a_chunk_is_split_on_whitespace():
  sentence = " ".join(["word"] * 200) + "."

  pieces = chunking.chunk_document(sentence, max_chars=300)

  assert len(pieces) > 1
  assert all(len(p.text) <= 300 for p in pieces[:-1])
  assert " ".join(p.text for p in pieces) == sentence


def test_kernel_trace_becomes_its_own_trace_chunks():
  before = "ipv6: prevent NULL dereference in ip6_output()."
  after = (
      "Fix this by checking the return value of ip6_dst_idev() before it is"
      " used, as the rest of the IPv6 output path already does everywhere."
  )

  pieces = chunking.chunk_document(f"{before} {_TRACE} {after}")

  assert [p.kind for p in pieces] == ["text", "trace", "text"]
  assert pieces[0].text == before
  assert pieces[2].text == after
  assert "ffffc9000927f0d8" in pieces[1].text


def test_long_trace_is_split_without_overlap():
  pieces = chunking.chunk_document(" ".join([_TRACE] * 6), max_chars=600)

  assert {p.kind for p in pieces} == {"trace"}
  assert all(len(p.text) <= 600 for p in pieces)
  assert " ".join(p.text for p in pieces) == " ".join([_TRACE] * 6)


@pytest.mark.parametrize(
    "kwargs", [{"max_chars": 0}, {"max_chars": -5}, {"overlap_sentences": -1}]
)
def test_invalid_limits_are_rejected(kwargs: dict[str, int]):
  with pytest.raises(ValueError):
    chunking.chunk_document("Some text.", **kwargs)
