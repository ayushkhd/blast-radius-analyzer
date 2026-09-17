# Evaluation question set

`questions.jsonl` is the saved question set that `make eval` scores every
retrieval configuration against. It was written once, against the reference
dataset, and is committed so that reruns are free and repeatable.
`results.json`, written beside it by `make eval`, holds the per-question
results of the last run.

## Format

One JSON object per line, with exactly these fields:

| Field          | Meaning                                                  |
|----------------|----------------------------------------------------------|
| `id`           | Stable id whose prefix follows the type, e.g. `para-11`  |
| `type`         | `identifier`, `paraphrase`, `product` or `negative`      |
| `query`        | What an analyst would type or paste                      |
| `gold_qids`    | QIDs that answer it, sorted as strings; `[]` if negative |
| `gold_cve_ids` | The CVE it was written from, if it was written from one  |
| `note`         | One line: what it is derived from and what it probes     |

Hosts attach to QIDs, so gold is a set of QIDs and the gold host set is derived
from it by SQL when a question is scored. `gold_cve_ids` lets retrieval be
scored at CVE level as well.

`blast_radius/evaluation/questions.py` loads the file and rejects malformed
lines, naming the file and line. `make eval` then checks every gold id against
the index artifact and refuses to score a set that does not match it.

## Composition

| Type         | Count | What it contains                                     |
|--------------|-------|------------------------------------------------------|
| `identifier` | 10    | CVE ids and QIDs, in the forms people paste them     |
| `paraphrase` | 20    | One write-up each, retold as an advisory or a chat   |
| `product`    | 5     | Product and version; gold spans several QIDs         |
| `negative`   | 5     | Absent from the corpus; the right answer is no match |

*   **Identifier.** Lower case, spaces or en dashes for hyphens, `QID-38913`,
    `qid:38919`, an id inside a sentence, and two ids in one query. They cover
    the one known-exploited CVE, a CVE inside the 219-CVE kernel bundle, the
    QID that has no CVE, one QID that has no write-up at all, and the one CVE
    that the scanner maps to two QIDs.
*   **Paraphrase.** 8 come from kernel CVEs: two from each of the large
    bundles and one from each of the other four kernel QIDs, each from a
    different subsystem. The other 12 cover OpenSSH (4), Apache httpd (3), the
    proxy (1), the single-QID findings (2) and the Ubuntu package updates (2).
    Queries carry no identifier, describe the product where that is natural
    ("a Go-based, cloud-native reverse proxy"), and run from one line to a
    four-sentence advisory. Each note names the near misses.
*   **Product.** Two OpenSSH cuts, one Apache httpd cut, and two queries
    without a version (the kernel updates, curl).
*   **Negative.** `CVE-2021-44228`, three well-known flaws in products the
    corpus never mentions, and one hard negative whose product is absent but
    whose text is about `sshd`.

## How gold was determined

Gold was derived from the two exports and checked by a one-off script, not by
eye. The checks that need only the artifact are repeated by `make eval` on
every run.

*   **Identifier.** Gold is the QID named, or every QID the scanner maps the
    named CVE to. The repository's own identifier extractor was run on each
    query to confirm that it finds exactly those ids.
*   **Paraphrase.** Each is written from one CVE description or one QID
    diagnosis, and gold is the QIDs of that source. For OpenSSH and Apache
    httpd the query describes the specific flaw, so that exactly one QID is
    right even though a dozen sibling diagnoses differ only in the flaw and
    the version range. The one exception is `para-04`: the scanner has two
    checks for that CVE, so both QIDs are gold. The script also rejects a
    query that names its product, uses one of the distinctive terms listed for
    its source, or copies five consecutive words from it.
*   **Product.** A QID is gold only if its write-up (the diagnosis or one of
    its CVE descriptions) names the product and states an affected range that
    lies entirely inside the range in the query. "OpenSSH 9.6 and below"
    therefore includes "prior to 7.6" and "up to version 9.6" and excludes
    "9.5 to 9.7", whose hosts may run 9.7. The stated bounds were transcribed
    into the script, each tied to a phrase that must occur in the write-up,
    and gold was computed from them. A query without a version takes every
    QID whose write-up is about the product.
*   **Negative.** The script confirms that the identifier and the product
    names occur nowhere in the corpus text.

## Adding a question

1.  Pick the source write-up and read its neighbours: the other QIDs of the
    same product, and for a kernel CVE the similar CVEs under other QIDs. Do
    not write a question that two QIDs answer equally well unless both are
    gold.
2.  Append one line with the next free id of its type. Keep `gold_qids` and
    `gold_cve_ids` sorted. For a CVE, list every QID it maps to.
3.  Say in `note` what the question is derived from, what it probes and which
    QIDs are the near misses.
4.  Run `make ingest` and `make eval`. The run stops with a list of problems
    if a gold id is missing from the artifact or a gold CVE maps to a QID that
    is not gold.
5.  A changed set changes every number in the results table, so rerun the
    table and say so in the commit.

## Known limits

*   **Synthetic.** The questions were written with a language model's help
    from the write-ups themselves, so they are probably easier than an
    analyst's. Rare terms were kept out of the paraphrases, but a long
    paraphrase still shares ordinary technical vocabulary with its source,
    which favours keyword search on a corpus of 560 documents.
*   **Small.** Forty questions give wide intervals: one question moves a rate
    by 0.025, and by 0.2 within the five product or negative questions. Read
    the counts, not only the rates.
*   **QID-level scoring of kernel paraphrases.** Hundreds of kernel CVEs are
    near-duplicates, and sibling CVEs under one QID resolve to the same hosts,
    so a kernel paraphrase counts as answered when its QID is found. CVE-level
    scores are reported beside it.
*   **Host sets overlap heavily.** All ten Apache httpd QIDs sit on one host,
    as do all 14 Ubuntu package QIDs outside the kernel, and the two largest
    OpenSSH QIDs share 333 hosts. Host precision and recall cannot separate
    sibling QIDs there; the QID-level metrics can.
*   **Product gold is a reading.** The inclusion rule is strict and stated
    above, but another defensible rule (any overlap with the queried range)
    would make every QID of the product gold.
