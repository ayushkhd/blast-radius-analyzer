"""Tests for blast_radius.verify."""

import pytest

from blast_radius import models
from blast_radius import verify

_DIAGNOSIS = models.ContextItem(
    id="c1",
    doc_type="qid",
    doc_id="90001",
    title="ExampleD Authentication Bypass",
    text=(
        "ExampleD is a remote login daemon.\n\nAffected Versions:\nExampleD up"
        " to version 2.3"
    ),
)
_PATCH = models.ContextItem(
    id="e1",
    doc_type="cve",
    doc_id="CVE-2024-0001",
    title="patch reference",
    text="Patch reference for CVE-2024-0001: https://example.test/patch/17",
)
_CONTEXT = [_DIAGNOSIS, _PATCH]


def _cite(quote: str, source_id: str = "c1") -> models.Citation:
  """Returns an unchecked citation of ``source_id``."""
  return models.Citation(source_id=source_id, quote=quote)


def _answer(
    *claims: models.Claim,
    summary: str = "ExampleD is affected.",
    caveats: tuple[str, ...] = (),
) -> models.Answer:
  """Returns an unchecked brief with the given claims."""
  return models.Answer(
      summary=summary, claims=list(claims), caveats=list(caveats)
  )


def _cited(text: str) -> models.Claim:
  """Returns a claim whose only citation is sound, so its text decides."""
  return models.Claim(text=text, citations=[_cite("remote login daemon")])


def test_verify_answer_verbatim_quote_verifies_the_claim():
  claim = models.Claim(
      text="ExampleD is a login daemon.",
      citations=[_cite("ExampleD is a remote login daemon.")],
  )

  checked, verification = verify.verify_answer(_answer(claim), _CONTEXT)

  assert checked.claims[0].verified is True
  assert checked.claims[0].citations[0].verified is True
  assert not checked.claims[0].problems
  assert verification.verified_claims == 1


def test_verify_answer_quote_differing_only_in_whitespace_verifies():
  claim = models.Claim(
      text="Versions up to 2.3 are affected.",
      citations=[_cite("Affected Versions: ExampleD  up to\nversion 2.3")],
  )

  checked, _ = verify.verify_answer(_answer(claim), _CONTEXT)

  assert checked.claims[0].verified is True


@pytest.mark.parametrize(
    "quote",
    [
        "ExampleD is a daemon for remote logins.",
        "exampled is a remote login daemon.",
        "ExampleD is a remote login daemon. ExampleD up to version 2.3",
        "ExampleD is a remote login daemon ... up to version 2.3",
    ],
    ids=["paraphrase", "other-case", "spliced", "ellipsis"],
)
def test_verify_answer_quote_that_is_not_in_the_source_fails(quote: str):
  claim = models.Claim(text="ExampleD is a daemon.", citations=[_cite(quote)])

  checked, verification = verify.verify_answer(_answer(claim), _CONTEXT)

  assert checked.claims[0].verified is False
  assert checked.claims[0].citations[0].verified is False
  assert checked.claims[0].problems == [
      "the quote attributed to c1 is not in it"
  ]
  assert verification.verified_claims == 0


def test_verify_answer_quote_from_another_source_fails():
  claim = models.Claim(
      text="A patch exists.",
      citations=[_cite("https://example.test/patch/17", source_id="c1")],
  )

  checked, _ = verify.verify_answer(_answer(claim), _CONTEXT)

  assert checked.claims[0].verified is False


def test_verify_answer_unknown_source_id_fails():
  claim = models.Claim(
      text="ExampleD is a daemon.",
      citations=[_cite("remote login daemon", source_id="c99")],
  )

  checked, _ = verify.verify_answer(_answer(claim), _CONTEXT)

  assert checked.claims[0].verified is False
  assert checked.claims[0].citations[0].verified is False
  assert checked.claims[0].problems == [
      "cites 'c99', which is not one of the sources"
  ]


@pytest.mark.parametrize("quote", ["", " \n "], ids=["empty", "blank"])
def test_verify_answer_empty_quote_fails(quote: str):
  claim = models.Claim(text="ExampleD is a daemon.", citations=[_cite(quote)])

  checked, _ = verify.verify_answer(_answer(claim), _CONTEXT)

  assert checked.claims[0].verified is False
  assert checked.claims[0].problems == ["quotes nothing from c1"]


def test_verify_answer_claim_without_citations_fails():
  claim = models.Claim(text="ExampleD is a daemon.")

  checked, verification = verify.verify_answer(_answer(claim), _CONTEXT)

  assert checked.claims[0].verified is False
  assert checked.claims[0].problems == ["cites no source"]
  assert verification.verified_claims == 0


def test_verify_answer_one_bad_citation_fails_the_claim_but_not_its_sibling():
  claim = models.Claim(
      text="ExampleD up to 2.3 is affected and has a patch.",
      citations=[
          _cite("ExampleD up to version 2.3"),
          _cite("the vendor has released version 2.4", source_id="e1"),
      ],
  )

  checked, _ = verify.verify_answer(_answer(claim), _CONTEXT)

  assert checked.claims[0].verified is False
  assert [c.verified for c in checked.claims[0].citations] == [True, False]


@pytest.mark.parametrize(
    "text, identifier",
    [
        ("It is the same flaw as cve-2099-4242.", "CVE-2099-4242"),
        ("The hosts also have QID 77777.", "QID 77777"),
        ("The host at 203.0.113.9 is exposed.", "203.0.113.9"),
        ("Patch I-0FEDCBA9876543210 first.", "i-0fedcba9876543210"),
        ("Patch vault-7 first.", "vault-7"),
    ],
    ids=["cve", "qid", "ip", "instance", "host"],
)
def test_verify_answer_identifier_the_model_was_not_shown_fails_the_claim(
    text: str, identifier: str
):
  checked, verification = verify.verify_answer(
      _answer(_cited(text)),
      _CONTEXT,
      shown_text="edge-proxy-a at 192.0.2.10 is i-0123456789abcdef0",
      inventory_names=["edge-proxy-a", "vault-7"],
  )

  assert checked.claims[0].verified is False
  assert checked.claims[0].citations[0].verified is True
  assert checked.claims[0].problems == [
      f"mentions {identifier}, which the model was not shown"
  ]
  assert verification.unknown_identifiers == [identifier]


def test_verify_answer_identifiers_found_only_in_shown_text_are_known():
  text = (
      "Patch edge-proxy-a (192.0.2.10, i-0123456789abcdef0) for QID 90002 and"
      " CVE-2024-0002."
  )

  checked, verification = verify.verify_answer(
      _answer(_cited(text)),
      _CONTEXT,
      shown_text=(
          "QID 90002 covers CVE-2024-0002. For example Edge-Proxy-A at"
          " 192.0.2.10, instance i-0123456789abcdef0."
      ),
      inventory_names=["edge-proxy-a"],
  )

  assert checked.claims[0].verified is True
  assert not verification.unknown_identifiers


def test_verify_answer_document_ids_and_titles_of_the_sources_are_known():
  titled = _DIAGNOSIS.model_copy(update={"title": "ExampleD on edge-proxy-a"})
  text = "QID 90001 and CVE-2024-0001 affect edge-proxy-a."

  checked, _ = verify.verify_answer(
      _answer(_cited(text)), [titled, _PATCH], inventory_names=["edge-proxy-a"]
  )

  assert checked.claims[0].verified is True


def test_verify_answer_host_name_inside_a_longer_name_is_not_a_mention():
  text = "Patch edge-proxy-a10 and edge-proxy-a.internal first."

  _, verification = verify.verify_answer(
      _answer(_cited(text)),
      _CONTEXT,
      shown_text="For example edge-proxy-a10, edge-proxy-a.internal",
      inventory_names=["edge-proxy-a", "edge-proxy-a10"],
  )

  assert not verification.unknown_identifiers


def test_verify_answer_host_name_at_the_end_of_a_sentence_is_a_mention():
  _, verification = verify.verify_answer(
      _answer(_cited("Start with edge-proxy-a.")),
      _CONTEXT,
      inventory_names=["edge-proxy-a"],
  )

  assert verification.unknown_identifiers == ["edge-proxy-a"]


def test_verify_answer_host_named_with_a_plain_word_is_not_looked_for():
  _, verification = verify.verify_answer(
      _answer(_cited("Patch the bastion hosts first.")),
      _CONTEXT,
      inventory_names=["bastion"],
  )

  assert not verification.unknown_identifiers


def test_verify_answer_count_after_a_qid_is_not_read_as_a_qid():
  text = "Nine hosts have QID 90001, 4 of them internet-facing."

  checked, _ = verify.verify_answer(_answer(_cited(text)), _CONTEXT)

  assert checked.claims[0].verified is True


def test_verify_answer_summary_with_an_unknown_identifier_is_reported():
  answer = _answer(
      _cited("ExampleD is a daemon."), summary="See also CVE-2099-4242."
  )

  checked, verification = verify.verify_answer(answer, _CONTEXT)

  assert verification.unknown_identifiers == ["CVE-2099-4242"]
  assert checked.claims[0].verified is True
  assert checked.summary == "See also CVE-2099-4242."


def test_verify_answer_caveat_with_an_unknown_identifier_is_reported():
  answer = _answer(
      _cited("ExampleD is a daemon."), caveats=("QID 77777 was not checked.",)
  )

  _, verification = verify.verify_answer(answer, _CONTEXT)

  assert verification.unknown_identifiers == ["QID 77777"]


def test_verify_answer_reports_each_unknown_identifier_once():
  answer = _answer(
      _cited("CVE-2099-4242 is related."),
      _cited("CVE-2099-4242 is worse than QID 77777."),
      summary="CVE-2099-4242 matters.",
  )

  _, verification = verify.verify_answer(answer, _CONTEXT)

  assert verification.unknown_identifiers == ["CVE-2099-4242", "QID 77777"]


def test_verify_answer_tallies_a_brief_whose_claims_all_verify():
  answer = _answer(
      _cited("ExampleD is a login daemon."),
      models.Claim(
          text="CVE-2024-0001 has a patch.",
          citations=[_cite("https://example.test/patch/17", source_id="e1")],
      ),
  )

  _, verification = verify.verify_answer(answer, _CONTEXT)

  assert verification == models.Verification(
      total_claims=2, verified_claims=2, unknown_identifiers=[]
  )


def test_verify_answer_keeps_failed_claims_in_place():
  answer = _answer(
      models.Claim(text="Nothing backs this up."),
      _cited("ExampleD is a login daemon."),
      _cited("CVE-2099-4242 is related."),
      caveats=("The data names no fixed version.",),
  )

  checked, verification = verify.verify_answer(answer, _CONTEXT)

  assert [claim.text for claim in checked.claims] == [
      "Nothing backs this up.",
      "ExampleD is a login daemon.",
      "CVE-2099-4242 is related.",
  ]
  assert [claim.verified for claim in checked.claims] == [False, True, False]
  assert checked.caveats == ["The data names no fixed version."]
  assert verification.total_claims == 3
  assert verification.verified_claims == 1


def test_verify_answer_leaves_the_answer_it_was_given_unchecked():
  answer = _answer(_cited("ExampleD is a login daemon."))

  verify.verify_answer(answer, _CONTEXT)

  assert answer.claims[0].verified is None
  assert answer.claims[0].citations[0].verified is None


def test_verify_answer_of_a_brief_without_claims_has_an_empty_tally():
  _, verification = verify.verify_answer(_answer(), _CONTEXT)

  assert verification == models.Verification(
      total_claims=0, verified_claims=0, unknown_identifiers=[]
  )
