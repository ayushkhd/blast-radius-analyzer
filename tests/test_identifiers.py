"""Tests for blast_radius.retrieval.identifiers."""

import dataclasses

import pytest

from blast_radius.retrieval import identifiers


def test_extract_finds_a_cve_id():
  found = identifiers.extract("CVE-2024-36971")

  assert found == identifiers.Identifiers(
      cve_ids=["CVE-2024-36971"], qids=[], remainder=""
  )


@pytest.mark.parametrize(
    "text",
    [
        "cve-2024-36971",
        "Cve-2024-36971",
        "cve 2024 36971",
        "CVE_2024_36971",
        "CVE\u20132024\u201336971",  # En dashes, as pasted from a PDF.
        "CVE\u20112024\u201136971",  # Non-breaking hyphens, from a document.
        "CVE-2024-\n36971",  # Wrapped across a line.
        "CVE - 2024 - 36971",
    ],
)
def test_extract_normalises_the_ways_a_cve_id_is_pasted(text: str):
  assert identifiers.extract(text).cve_ids == ["CVE-2024-36971"]


@pytest.mark.parametrize("number", ["1234", "12345", "123456", "1234567"])
def test_extract_accepts_cve_sequence_numbers_of_four_to_seven_digits(
    number: str,
):
  assert identifiers.extract(f"CVE-2021-{number}").cve_ids == [
      f"CVE-2021-{number}"
  ]


@pytest.mark.parametrize(
    "text",
    [
        "CVE-2024-123",  # Sequence number too short.
        "CVE-2024-12345678",  # Too long: not to be read as its first seven.
        "CVE-24-36971",  # Two-digit year.
        "XCVE-2024-36971",
        "CVE-2024-36971abc",
        "CVE-2024",
    ],
)
def test_extract_leaves_a_malformed_cve_id_in_the_remainder(text: str):
  found = identifiers.extract(text)

  assert not found.cve_ids
  assert found.remainder == text


def test_extract_keeps_cve_ids_in_order_without_repeats():
  text = "CVE-2024-36971, CVE-2023-48795 and cve_2024_36971"

  assert identifiers.extract(text).cve_ids == [
      "CVE-2024-36971",
      "CVE-2023-48795",
  ]


@pytest.mark.parametrize(
    "text",
    [
        "QID 38919",
        "qid 38919",
        "qid:38919",
        "QID: 38919",
        "QID-38919",
        "QID#38919",
        "QID #38919",
        "QID38919",
    ],
)
def test_extract_finds_a_qid_written_with_the_keyword(text: str):
  found = identifiers.extract(text)

  assert found.qids == ["38919"]
  assert found.remainder == ""


@pytest.mark.parametrize(
    "text",
    [
        "QIDs 38919, 38913 and 105936",
        "QIDs 38919,38913,105936",
        "qids 38919, 38913, and 105936",
        "QID 38919 or 38913 & 105936",
    ],
)
def test_extract_reads_a_list_of_qids_after_one_keyword(text: str):
  found = identifiers.extract(text)

  assert found.qids == ["38919", "38913", "105936"]
  assert found.remainder == ""


@pytest.mark.parametrize(
    "text",
    [
        "38919",
        "port 22 and 443",
        "squid 3128",  # "qid" inside another word is not the keyword.
        "QID 38919abc",
        "what is a QID?",
    ],
)
def test_extract_never_reads_a_bare_number_as_a_qid(text: str):
  found = identifiers.extract(text)

  assert not found.qids
  assert found.remainder == text


def test_extract_takes_a_number_after_and_for_another_qid():
  found = identifiers.extract("QID 38919 and 22 hosts")

  assert found.qids == ["38919", "22"]
  assert found.remainder == "hosts"


def test_extract_keeps_qids_in_order_without_repeats():
  text = "QID 38919, QIDs 105936 and 38919"

  assert identifiers.extract(text).qids == ["38919", "105936"]


def test_extract_separates_identifiers_from_the_text_to_search():
  text = "OpenSSH auth bypass CVE-2023-48795, seen as QID 38913 on bastions"

  found = identifiers.extract(text)

  assert found.cve_ids == ["CVE-2023-48795"]
  assert found.qids == ["38913"]
  assert found.remainder == "OpenSSH auth bypass , seen as on bastions"


def test_extract_tidies_the_whitespace_of_the_remainder():
  found = identifiers.extract("  OpenSSH \n\n auth\tbypass  ")

  assert found.remainder == "OpenSSH auth bypass"


@pytest.mark.parametrize(
    "text",
    [
        "CVE-2023-48795, QID 38913.",
        "(CVE-2023-48795)",
        "CVE-2023-48795 and QID 38913",
        "CVE-2023-48795 or CVE-2024-36971?",
        "",
        " \n ",
        "?!",
    ],
)
def test_extract_remainder_is_empty_when_nothing_is_left_to_search(text: str):
  assert identifiers.extract(text).remainder == ""


def test_extract_of_plain_text_finds_no_identifiers():
  found = identifiers.extract("kernel use after free in netfilter")

  assert found == identifiers.Identifiers(
      cve_ids=[], qids=[], remainder="kernel use after free in netfilter"
  )


def test_extract_long_runs_of_separators_do_not_stall_the_patterns():
  # A pattern that can split a run of separators in more than one way tries
  # every split when the match fails after the run. Written that way, the
  # QID list pattern stalls for minutes on twenty commas.
  text = "CVE" + " " * 20000 + "QID 1" + " , " * 20000 + "x"

  found = identifiers.extract(text)

  assert found == identifiers.Identifiers(
      cve_ids=[], qids=["1"], remainder="CVE " + ", " * 20000 + "x"
  )


def test_identifiers_is_immutable():
  found = identifiers.extract("QID 38919")

  with pytest.raises(dataclasses.FrozenInstanceError):
    found.remainder = "changed"  # type: ignore[misc]
