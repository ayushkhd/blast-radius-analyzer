"""Tests for blast_radius.textproc."""

from blast_radius import textproc

_TRACE = (
    "RIP: 0010:nft_setelem_data_deactivate+0xe4/0xf0 [nf_tables]"
    " Code: 83 f8 01 77 ab 49 8d 7c 24 08 e8 37 5e d0 de 49 8b 1c 24"
    " RSP: 0018:ffff888140f9f4b0 EFLAGS: 00010246"
    " RAX: 0000000000000000 RBX: ffff8881434f5288 RCX: dffffc0000000000"
    " Call Trace: <TASK> ? __warn+0x9f/0x1a0"
    " ? nft_setelem_data_deactivate+0xe4/0xf0 [nf_tables]"
    " nft_mapelem_deactivate+0x24/0x30 [nf_tables]"
    " nft_rhash_walk+0xdd/0x180 [nf_tables] </TASK>"
)


def test_strip_html_turns_block_tags_into_paragraph_breaks():
  diagnosis = (
      "OpenSSH is a tool.<P>\n\nAffected Versions:<BR>\nOpenSSH up to version"
      " 9.6<P>"
  )

  assert textproc.strip_html(diagnosis) == (
      "OpenSSH is a tool.\n\nAffected Versions:\n\nOpenSSH up to version 9.6"
  )


def test_strip_html_decodes_entities_and_drops_inline_tags():
  text = "uses <b>&quot;dpkg&quot;</b> &amp; friends&nbsp;here"

  assert textproc.strip_html(text) == 'uses "dpkg" & friends here'


def test_strip_html_leaves_plain_text_alone():
  assert textproc.strip_html("a < b and c > d") == "a < b and c > d"


def test_remove_kernel_boilerplate_removes_only_the_opening_sentence():
  text = (
      "In the Linux kernel, the following vulnerability has been resolved: "
      " nfs: handle error of rpc_proc_register()."
  )

  assert textproc.remove_kernel_boilerplate(text) == (
      "nfs: handle error of rpc_proc_register()."
  )
  assert textproc.remove_kernel_boilerplate("Traefik is a proxy.") == (
      "Traefik is a proxy."
  )


def test_normalise_whitespace_collapses_runs_and_trims():
  assert textproc.normalise_whitespace("  a\n\n b\t c ") == "a b c"


def test_split_sentences_splits_on_punctuation_and_paragraphs():
  text = "First sentence. Second one!\n\nA new paragraph without a full stop"

  assert textproc.split_sentences(text) == [
      "First sentence.",
      "Second one!",
      "A new paragraph without a full stop",
  ]


def test_split_sentences_keeps_abbreviations_and_versions_together():
  text = "Fixed in 2.4.55, e.g. by upgrading. See the advisory."

  assert textproc.split_sentences(text) == [
      "Fixed in 2.4.55, e.g. by upgrading.",
      "See the advisory.",
  ]


def test_split_sentences_of_empty_text_is_empty():
  assert not textproc.split_sentences(" \n\n ")


def test_segment_traces_returns_one_prose_segment_for_plain_text():
  text = "Traefik is a proxy.\n\nIt can be made to crash by a bad header."

  assert textproc.segment_traces(text) == [(False, text)]


def test_segment_traces_of_empty_text_is_empty():
  assert not textproc.segment_traces("")


def test_segment_traces_separates_a_trace_from_the_prose_around_it():
  before = "netfilter: restore set elements when delete set fails."
  after = (
      "Fix this by restoring the set elements from the abort path, which is"
      " what the commit path already does for the same set type and flags."
  )

  segments = textproc.segment_traces(f"{before} {_TRACE} {after}")

  assert [is_trace for is_trace, _ in segments] == [False, True, False]
  assert segments[0][1] == before
  assert segments[1][1].startswith("RIP: 0010:nft_setelem")
  assert segments[1][1].endswith("</TASK>")
  assert segments[2][1] == after


def test_segment_traces_ignores_a_single_address_in_a_sentence():
  text = (
      "The buffer at ffff888030652f00 is freed twice when the device is"
      " removed while a transfer is still in flight, which corrupts memory."
  )

  assert textproc.segment_traces(text) == [(False, text)]


def test_segment_traces_does_not_mistake_short_words_for_opcode_bytes():
  text = (
      "It must be added so that an ad hoc fd can be de-registered and a bad"
      " cb be fed to the API, as it would be for an fe or a be device."
  )

  assert textproc.segment_traces(text) == [(False, text)]


def test_segment_traces_folds_a_log_fragment_between_two_traces():
  text = f"Summary of the bug. {_TRACE} Allocated by task 506: {_TRACE}"

  segments = textproc.segment_traces(text)

  assert [is_trace for is_trace, _ in segments] == [False, True]
  assert "Allocated by task 506:" in segments[1][1]
