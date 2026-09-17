"""Tests for blast_radius.ranking."""

import dataclasses
import itertools
import os

import pytest

from blast_radius import config
from blast_radius import models
from blast_radius import ranking


@pytest.fixture(autouse=True)
def _no_blast_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  """Keeps ``BLAST_*`` variables set on this machine out of the settings."""
  for name in list(os.environ):
    if name.upper().startswith("BLAST_"):
      monkeypatch.delenv(name)


def _settings(**overrides: float) -> config.Settings:
  """Returns the default settings with ``overrides``, reading no ``.env``."""
  return config.Settings(_env_file=None, **overrides)


def _host(
    host_id: str,
    name: str = "app",
    *,
    criticality: int = 3,
    internet_facing: bool = False,
    state: str = models.STATE_RUNNING,
    security_group: str | None = None,
    cluster: str | None = None,
) -> models.Host:
  """Returns a running, internal, criticality-3 host unless told otherwise."""
  return models.Host(
      id=host_id,
      name=name,
      criticality=criticality,
      state=state,
      internet_facing=internet_facing,
      security_group=security_group,
      cluster=cluster,
  )


def _cve(
    cve_id: str = "CVE-2024-0001",
    *,
    cvss: float | None = None,
    epss_percentile: float | None = None,
    known_exploited: bool = False,
    cogent_risk_score: float | None = None,
) -> models.CveSummary:
  """Returns a CVE that carries no signal unless told otherwise."""
  return models.CveSummary(
      cve_id=cve_id,
      cvss=cvss,
      epss_percentile=epss_percentile,
      known_exploited=known_exploited,
      cogent_risk_score=cogent_risk_score,
  )


def _match(
    qid: str, *cves: models.CveSummary, severity: int | None = None
) -> models.QidMatch:
  """Returns a matched QID with ``cves`` and the Qualys ``severity``."""
  return models.QidMatch(
      qid=qid, severity=severity, matched_by="search", cves=list(cves)
  )


def _ranked(host: models.Host, priority: float) -> models.RankedHost:
  """Returns ``host`` as ``rank_hosts`` would, at a priority chosen freely.

  Grouping reads the host and the priority only, so the factors are filler.

  Args:
    host: The host.
    priority: The priority to give it.
  """
  factors = models.PriorityFactors(
      severity=0.0,
      epss_percentile=0.0,
      known_exploited=False,
      threat=priority,
      internet_facing=host.internet_facing,
      criticality=host.criticality,
      exposure=1.0,
      driving_qid="100",
  )
  return models.RankedHost(
      host=host, qids=["100"], priority=priority, factors=factors
  )


def _ids(ranked: list[models.RankedHost]) -> list[str]:
  """Returns the host ids of ``ranked``, in order."""
  return [ranked_host.host.id for ranked_host in ranked]


# ---------------------------------------------------------------------------
# qid_threat
# ---------------------------------------------------------------------------


def test_qid_threat_with_cvss_takes_the_highest_score_over_ten():
  match = _match(
      "100",
      _cve("CVE-2024-0001", cvss=7.5),
      _cve("CVE-2024-0002", cvss=9.8),
      severity=2,
  )

  threat = ranking.qid_threat(match, _settings())

  assert threat.severity == pytest.approx(0.98)


def test_qid_threat_with_cvss_on_one_cve_ignores_the_qualys_severity():
  match = _match(
      "100",
      _cve("CVE-2024-0001", cvss=5.0),
      _cve("CVE-2024-0002"),
      severity=5,
  )

  threat = ranking.qid_threat(match, _settings())

  assert threat.severity == pytest.approx(0.5)


def test_qid_threat_without_any_cvss_falls_back_to_the_qualys_severity():
  match = _match(
      "100", _cve("CVE-2024-0001"), _cve("CVE-2024-0002"), severity=4
  )

  threat = ranking.qid_threat(match, _settings())

  assert threat.severity == pytest.approx(0.8)


def test_qid_threat_without_cves_scores_the_qualys_severity_alone():
  match = _match("100", severity=4)

  threat = ranking.qid_threat(match, _settings())

  assert dataclasses.asdict(threat) == pytest.approx(
      {
          "qid": "100",
          "severity": 0.8,
          "epss_percentile": 0.0,
          "known_exploited": False,
          "threat": 0.4,  # 0.5 * 0.8
          "cogent_risk_score": None,
      }
  )


def test_qid_threat_without_cvss_or_qualys_severity_is_zero():
  match = _match("100", _cve())

  threat = ranking.qid_threat(match, _settings())

  assert threat.severity == 0.0
  assert threat.threat == 0.0


def test_qid_threat_takes_each_maximum_separately_over_the_cves():
  match = _match(
      "100",
      _cve("CVE-2024-0001", cvss=9.8, epss_percentile=0.1),
      _cve("CVE-2024-0002", cvss=4.0, epss_percentile=0.95),
      _cve("CVE-2024-0003"),
  )

  threat = ranking.qid_threat(match, _settings())

  assert threat.severity == pytest.approx(0.98)
  assert threat.epss_percentile == pytest.approx(0.95)


def test_qid_threat_with_one_known_exploited_cve_is_known_exploited():
  match = _match(
      "100",
      _cve("CVE-2024-0001"),
      _cve("CVE-2024-0002", known_exploited=True),
  )

  threat = ranking.qid_threat(match, _settings())

  assert threat.known_exploited
  assert threat.threat == pytest.approx(0.2)


def test_qid_threat_blends_the_signals_with_the_default_weights():
  match = _match(
      "100", _cve(cvss=8.0, epss_percentile=0.5, known_exploited=True)
  )

  threat = ranking.qid_threat(match, _settings())

  # 0.5 * 0.8 + 0.3 * 0.5 + 0.2 * 1
  assert threat.threat == pytest.approx(0.75)


def test_qid_threat_uses_the_configured_weights_without_renormalising():
  settings = _settings(
      weight_severity=1.0, weight_epss=2.0, weight_known_exploited=3.0
  )
  match = _match(
      "100", _cve(cvss=5.0, epss_percentile=0.5, known_exploited=True)
  )

  threat = ranking.qid_threat(match, settings)

  # 1 * 0.5 + 2 * 0.5 + 3 * 1
  assert threat.threat == pytest.approx(4.5)


@pytest.mark.parametrize(
    ("cvss", "expected"),
    [(12.0, 1.0), (-3.0, 0.0), (float("nan"), 0.0)],
)
def test_qid_threat_clamps_a_cvss_score_outside_its_scale(
    cvss: float, expected: float
):
  match = _match("100", _cve(cvss=cvss), severity=3)

  threat = ranking.qid_threat(match, _settings())

  assert threat.severity == expected


@pytest.mark.parametrize(
    ("epss_percentile", "expected"),
    [(1.7, 1.0), (-0.2, 0.0), (float("nan"), 0.0)],
)
def test_qid_threat_clamps_an_epss_percentile_outside_its_scale(
    epss_percentile: float, expected: float
):
  match = _match("100", _cve(epss_percentile=epss_percentile))

  threat = ranking.qid_threat(match, _settings())

  assert threat.epss_percentile == expected


@pytest.mark.parametrize(("severity", "expected"), [(9, 1.0), (-1, 0.0)])
def test_qid_threat_clamps_a_qualys_severity_outside_its_scale(
    severity: int, expected: float
):
  match = _match("100", severity=severity)

  threat = ranking.qid_threat(match, _settings())

  assert threat.severity == expected


def test_qid_threat_reports_the_highest_export_risk_score_without_using_it():
  unscored = _match("100", _cve(cvss=7.0, epss_percentile=0.4))
  scored = _match(
      "100",
      _cve("CVE-2024-0001", cvss=7.0, epss_percentile=0.4),
      _cve("CVE-2024-0002", cogent_risk_score=3.1),
      _cve("CVE-2024-0003", cogent_risk_score=9.9),
  )

  without_score = ranking.qid_threat(unscored, _settings())
  with_score = ranking.qid_threat(scored, _settings())

  assert without_score.cogent_risk_score is None
  assert with_score.cogent_risk_score == 9.9
  assert with_score.threat == without_score.threat


def test_qid_threat_result_is_immutable():
  threat = ranking.qid_threat(_match("100", severity=3), _settings())

  with pytest.raises(dataclasses.FrozenInstanceError):
    threat.threat = 1.0


# ---------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------


def test_exposure_of_an_internal_host_at_baseline_criticality_is_one():
  host = _host("h1", criticality=3)

  assert ranking.exposure(host, _settings()) == pytest.approx(1.0)


def test_exposure_of_an_internet_facing_host_doubles_its_criticality_ratio():
  host = _host("h1", criticality=5, internet_facing=True)

  # 2 * (5 / 3)
  assert ranking.exposure(host, _settings()) == pytest.approx(10 / 3)


def test_exposure_below_baseline_criticality_is_less_than_one():
  host = _host("h1", criticality=1)

  assert ranking.exposure(host, _settings()) == pytest.approx(1 / 3)


def test_exposure_uses_the_configured_multiplier_and_baseline():
  settings = _settings(internet_facing_multiplier=3.0, baseline_criticality=5)
  host = _host("h1", criticality=4, internet_facing=True)

  # 3 * (4 / 5)
  assert ranking.exposure(host, settings) == pytest.approx(2.4)


# ---------------------------------------------------------------------------
# rank_hosts
# ---------------------------------------------------------------------------


def test_rank_hosts_puts_an_internet_facing_critical_host_first():
  worker = _host("h1", "worker")
  edge = _host("h2", "edge", criticality=5, internet_facing=True)
  affected = [(worker, ["100"]), (edge, ["100"])]

  ranked, inactive = ranking.rank_hosts(
      affected, [_match("100", severity=4)], _settings()
  )

  assert _ids(ranked) == ["h2", "h1"]
  # A threat of 0.4 times exposures of 2 * (5 / 3) and 1.
  assert [r.priority for r in ranked] == pytest.approx([4 / 3, 0.4])
  assert not inactive


def test_rank_hosts_fills_every_factor_of_a_ranked_host():
  host = _host("h1", "edge", criticality=5, internet_facing=True)
  cve = _cve(
      cvss=9.8, epss_percentile=0.9, known_exploited=True, cogent_risk_score=8.8
  )

  ranked, _ = ranking.rank_hosts(
      [(host, ["100"])], [_match("100", cve, severity=3)], _settings()
  )

  assert ranked[0].host == host
  assert ranked[0].qids == ["100"]
  assert ranked[0].priority == pytest.approx(3.2)  # 0.96 * 2 * (5 / 3)
  assert ranked[0].factors.model_dump() == pytest.approx(
      {
          "severity": 0.98,
          "epss_percentile": 0.9,
          "known_exploited": True,
          "threat": 0.96,  # 0.5 * 0.98 + 0.3 * 0.9 + 0.2
          "internet_facing": True,
          "criticality": 5,
          "exposure": 10 / 3,
          "driving_qid": "100",
          "cogent_risk_score": 8.8,
      }
  )


def test_rank_hosts_with_several_qids_is_driven_by_the_highest_threat():
  mild = _match("100", severity=2)
  severe = _match("200", _cve(cvss=9.0, epss_percentile=0.8))

  ranked, _ = ranking.rank_hosts(
      [(_host("h1"), ["100", "200"])], [mild, severe], _settings()
  )

  assert ranked[0].factors.driving_qid == "200"
  assert ranked[0].factors.threat == pytest.approx(0.69)  # 0.45 + 0.24


def test_rank_hosts_keeps_the_qids_of_a_host_in_the_order_given():
  matches = [_match("100", severity=2), _match("200", severity=4)]

  ranked, _ = ranking.rank_hosts(
      [(_host("h1"), ["200", "100"])], matches, _settings()
  )

  assert ranked[0].qids == ["200", "100"]


def test_rank_hosts_scores_a_host_by_its_own_qids_only():
  mild = _match("100", severity=2)
  severe = _match("200", _cve(cvss=9.0, epss_percentile=0.8))
  affected = [(_host("h1"), ["100"]), (_host("h2"), ["100", "200"])]

  ranked, _ = ranking.rank_hosts(affected, [mild, severe], _settings())

  assert _ids(ranked) == ["h2", "h1"]
  assert ranked[1].factors.driving_qid == "100"
  assert ranked[1].priority == pytest.approx(0.2)


@pytest.mark.parametrize("qids", [["9", "100"], ["100", "9"]])
def test_rank_hosts_with_tied_qids_is_driven_by_the_first_in_string_order(
    qids: list[str],
):
  matches = [_match("9", severity=4), _match("100", severity=4)]

  ranked, _ = ranking.rank_hosts([(_host("h1"), qids)], matches, _settings())

  assert ranked[0].factors.driving_qid == "100"


def test_rank_hosts_breaks_priority_ties_by_name_then_id():
  affected = [
      (_host("h3", "alpha"), ["100"]),
      (_host("h1", "beta"), ["100"]),
      (_host("h2", "alpha"), ["100"]),
  ]

  ranked, _ = ranking.rank_hosts(
      affected, [_match("100", severity=4)], _settings()
  )

  assert _ids(ranked) == ["h2", "h3", "h1"]


def test_rank_hosts_gives_the_same_result_for_every_input_order():
  affected = [
      (_host("h1", "beta"), ["100"]),
      (_host("h2", "alpha"), ["100"]),
      (_host("h3", "alpha"), ["100"]),
      (_host("h4", "gone", state="TERMINATED"), ["100"]),
      (_host("h5", "gone", state="TERMINATED"), ["100"]),
  ]
  matches = [_match("100", severity=4)]
  expected = ranking.rank_hosts(affected, matches, _settings())

  results = [
      ranking.rank_hosts(list(permutation), matches, _settings())
      for permutation in itertools.permutations(affected)
  ]

  assert all(result == expected for result in results)


def test_rank_hosts_sets_hosts_that_are_not_running_aside_unranked():
  running = _host("h1", "web")
  terminated = _host(
      "h2", "old-web", criticality=5, internet_facing=True, state="TERMINATED"
  )
  affected = [(terminated, ["100"]), (running, ["100"])]

  ranked, inactive = ranking.rank_hosts(
      affected, [_match("100", severity=4)], _settings()
  )

  assert _ids(ranked) == ["h1"]
  assert inactive == [terminated]


def test_rank_hosts_sorts_inactive_hosts_by_name_then_id():
  affected = [
      (_host("h3", "beta", state="STOPPED"), ["100"]),
      (_host("h2", "alpha", state="TERMINATED"), ["100"]),
      (_host("h1", "beta", state="SHUTTING_DOWN"), ["100"]),
  ]

  _, inactive = ranking.rank_hosts(
      affected, [_match("100", severity=4)], _settings()
  )

  assert [host.id for host in inactive] == ["h2", "h1", "h3"]


def test_rank_hosts_with_custom_weights_reorders_the_hosts():
  severe = _match("100", _cve("CVE-2024-0001", cvss=9.0, epss_percentile=0.1))
  likely = _match("200", _cve("CVE-2024-0002", cvss=4.0, epss_percentile=0.6))
  affected = [(_host("h1"), ["100"]), (_host("h2"), ["200"])]
  epss_heavy = _settings(
      weight_severity=0.1, weight_epss=0.9, weight_known_exploited=0.0
  )

  by_default, _ = ranking.rank_hosts(affected, [severe, likely], _settings())
  by_epss, _ = ranking.rank_hosts(affected, [severe, likely], epss_heavy)

  assert _ids(by_default) == ["h1", "h2"]  # 0.48 against 0.38
  assert _ids(by_epss) == ["h2", "h1"]  # 0.18 against 0.58


def test_rank_hosts_without_affected_hosts_returns_two_empty_lists():
  assert ranking.rank_hosts([], [_match("100", severity=4)], _settings()) == (
      [],
      [],
  )


def test_rank_hosts_with_a_qid_missing_from_the_matches_raises():
  affected = [(_host("h1"), ["100", "999"])]

  with pytest.raises(ValueError, match="h1.*999"):
    ranking.rank_hosts(affected, [_match("100", severity=4)], _settings())


def test_rank_hosts_with_an_unmatched_qid_on_an_inactive_host_raises():
  affected = [(_host("h1", state="TERMINATED"), ["999"])]

  with pytest.raises(ValueError, match="h1.*999"):
    ranking.rank_hosts(affected, [_match("100", severity=4)], _settings())


def test_rank_hosts_with_a_host_affected_by_no_qid_raises():
  with pytest.raises(ValueError, match="h1"):
    ranking.rank_hosts(
        [(_host("h1"), [])], [_match("100", severity=4)], _settings()
    )


# ---------------------------------------------------------------------------
# name_pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "ip-10-1-2-3.eu-west-1.compute.internal",
            "ip-#.eu-west-#.compute.internal",
        ),
        ("10.1.2.3", "#"),
        ("build-agent-0a1b2c3d", "build-agent-#"),
        ("BUILD-AGENT-0A1B2C3D", "BUILD-AGENT-#"),
        ("i-0a1b2c3d4e5f67890", "i-#"),
        ("cache_deadbeef.internal", "cache_#.internal"),
        ("web01", "web#"),
        ("node-2-rack-14", "node-#-rack-#"),
        ("bastion", "bastion"),
        ("", ""),
    ],
)
def test_name_pattern_replaces_the_parts_that_vary(name: str, expected: str):
  assert ranking.name_pattern(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Hex characters only, but too short to be an id.
        ("db01", "db#"),
        ("facade-proxy", "facade-proxy"),
        # Long enough, but not a whole token of hex characters.
        ("worker-0a1b2c3dxyz", "worker-#a#b#c#dxyz"),
        # Four groups of digits, but the first is too long for an address.
        ("1234-1-2-3", "#-#-#-#"),
    ],
)
def test_name_pattern_leaves_lookalikes_to_the_digit_rule(
    name: str, expected: str
):
  assert ranking.name_pattern(name) == expected


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            "ip-10-1-2-3.eu-west-1.compute.internal",
            "ip-10-9-8-7.eu-west-1.compute.internal",
        ),
        ("build-agent-0a1b2c3d", "build-agent-99ffee00"),
        ("web01", "web02"),
    ],
)
def test_name_pattern_is_shared_by_names_that_differ_only_in_ids(
    first: str, second: str
):
  assert ranking.name_pattern(first) == ranking.name_pattern(second)


def test_name_pattern_differs_for_names_that_differ_in_words():
  assert ranking.name_pattern("web01") != ranking.name_pattern("db01")


# ---------------------------------------------------------------------------
# group_hosts
# ---------------------------------------------------------------------------


def test_group_hosts_folds_hosts_that_share_a_security_group():
  ranked = [
      _ranked(_host("h1", "worker-a", security_group="workers-sg"), 0.4),
      _ranked(_host("h2", "worker-b", security_group="workers-sg"), 0.4),
  ]

  groups = ranking.group_hosts(ranked, _settings())

  assert groups == [
      models.HostGroup(
          key="sg:workers-sg",
          label="workers-sg",
          count=2,
          priority=0.4,
          internet_facing_count=0,
          example_hosts=["worker-a", "worker-b"],
          host_ids=["h1", "h2"],
      )
  ]


def test_group_hosts_prefers_the_security_group_to_the_cluster():
  ranked = [
      _ranked(_host("h1", security_group="workers-sg", cluster="c1"), 0.4)
  ]

  groups = ranking.group_hosts(ranked, _settings())

  assert [group.key for group in groups] == ["sg:workers-sg"]


def test_group_hosts_without_a_security_group_falls_back_to_the_cluster():
  ranked = [
      _ranked(_host("h1", "node-a", cluster="c1"), 0.4),
      _ranked(_host("h2", "node-b", cluster="c1"), 0.4),
      _ranked(_host("h3", "node-c", cluster="c2"), 0.4),
  ]

  groups = ranking.group_hosts(ranked, _settings())

  assert [(g.key, g.label, g.count) for g in groups] == [
      ("cluster:c1", "c1", 2),
      ("cluster:c2", "c2", 1),
  ]


def test_group_hosts_without_group_or_cluster_falls_back_to_the_name_pattern():
  ranked = [
      _ranked(_host("h1", "build-agent-0a1b2c3d"), 0.4),
      _ranked(_host("h2", "build-agent-99ffee00"), 0.4),
      _ranked(_host("h3", "bastion"), 0.4),
  ]

  groups = ranking.group_hosts(ranked, _settings())

  assert [(g.key, g.label, g.count) for g in groups] == [
      ("name:build-agent-#", "build-agent-#", 2),
      ("name:bastion", "bastion", 1),
  ]


def test_group_hosts_treats_an_empty_group_and_cluster_as_missing():
  ranked = [_ranked(_host("h1", "web01", security_group="", cluster=""), 0.4)]

  groups = ranking.group_hosts(ranked, _settings())

  assert [group.key for group in groups] == ["name:web#"]


def test_group_hosts_keeps_a_comma_separated_security_group_whole():
  ranked = [
      _ranked(_host("h1", security_group="default,edge-sg"), 0.4),
      _ranked(_host("h2", security_group="default,edge-sg"), 0.4),
      _ranked(_host("h3", security_group="default"), 0.4),
      _ranked(_host("h4", security_group="edge-sg,default"), 0.4),
  ]

  groups = ranking.group_hosts(ranked, _settings())

  assert [(g.key, g.label, g.host_ids) for g in groups] == [
      ("sg:default,edge-sg", "default,edge-sg", ["h1", "h2"]),
      ("sg:default", "default", ["h3"]),
      ("sg:edge-sg,default", "edge-sg,default", ["h4"]),
  ]


def test_group_hosts_keeps_equal_labels_under_different_keys_apart():
  ranked = [
      _ranked(_host("h1", security_group="prod"), 0.4),
      _ranked(_host("h2", cluster="prod"), 0.4),
  ]

  groups = ranking.group_hosts(ranked, _settings())

  assert [(g.key, g.label) for g in groups] == [
      ("cluster:prod", "prod"),
      ("sg:prod", "prod"),
  ]


def test_group_hosts_takes_the_priority_of_its_most_urgent_host():
  ranked = [
      _ranked(_host("h1", security_group="edge-sg"), 0.4),
      _ranked(_host("h2", security_group="edge-sg"), 1.2),
      _ranked(_host("h3", security_group="edge-sg"), 0.7),
  ]

  groups = ranking.group_hosts(ranked, _settings())

  assert groups[0].priority == 1.2


def test_group_hosts_counts_the_internet_facing_hosts():
  ranked = [
      _ranked(_host("h1", security_group="edge-sg", internet_facing=True), 0.8),
      _ranked(_host("h2", security_group="edge-sg", internet_facing=True), 0.8),
      _ranked(_host("h3", security_group="edge-sg"), 0.4),
  ]

  groups = ranking.group_hosts(ranked, _settings())

  assert groups[0].count == 3
  assert groups[0].internet_facing_count == 2


def test_group_hosts_lists_hosts_in_priority_order_whatever_the_input_order():
  ranked = [
      _ranked(_host("h1", "zeta", security_group="edge-sg"), 0.4),
      _ranked(_host("h2", "beta", security_group="edge-sg"), 0.4),
      _ranked(_host("h3", "alpha", security_group="edge-sg"), 0.4),
      _ranked(_host("h4", "omega", security_group="edge-sg"), 1.2),
  ]

  groups = ranking.group_hosts(ranked, _settings())

  assert groups[0].host_ids == ["h4", "h3", "h2", "h1"]
  assert groups[0].example_hosts == ["omega", "alpha", "beta", "zeta"]


def test_group_hosts_caps_the_examples_at_the_configured_count():
  ranked = [
      _ranked(_host(f"h{i}", f"worker-{i}", security_group="workers-sg"), 0.4)
      for i in range(1, 7)
  ]

  by_default = ranking.group_hosts(ranked, _settings())
  capped = ranking.group_hosts(ranked, _settings(group_example_count=2))

  assert by_default[0].example_hosts == [
      "worker-1",
      "worker-2",
      "worker-3",
      "worker-4",
  ]
  assert capped[0].example_hosts == ["worker-1", "worker-2"]
  assert capped[0].count == 6
  assert len(capped[0].host_ids) == 6


def test_group_hosts_lists_a_name_shared_by_several_hosts_once():
  ranked = [
      _ranked(_host("h1", "worker", security_group="workers-sg"), 0.4),
      _ranked(_host("h2", "worker", security_group="workers-sg"), 0.4),
      _ranked(_host("h3", "worker", security_group="workers-sg"), 0.4),
      _ranked(_host("h4", "worker-spare", security_group="workers-sg"), 0.4),
  ]

  groups = ranking.group_hosts(ranked, _settings(group_example_count=2))

  assert groups[0].example_hosts == ["worker", "worker-spare"]
  assert groups[0].host_ids == ["h1", "h2", "h3", "h4"]


def test_group_hosts_orders_groups_by_priority_then_count_then_label():
  ranked = [
      _ranked(_host("h1", security_group="small-b"), 0.4),
      _ranked(_host("h2", security_group="small-a"), 0.4),
      _ranked(_host("h3", security_group="large"), 0.4),
      _ranked(_host("h4", security_group="large"), 0.4),
      _ranked(_host("h5", security_group="urgent"), 1.2),
  ]

  groups = ranking.group_hosts(ranked, _settings())

  assert [group.label for group in groups] == [
      "urgent",
      "large",
      "small-a",
      "small-b",
  ]


def test_group_hosts_gives_the_same_groups_for_every_input_order():
  ranked = [
      _ranked(_host("h1", "edge", security_group="edge-sg"), 1.2),
      _ranked(_host("h2", "node", cluster="c1"), 0.4),
      _ranked(_host("h3", "node", cluster="c1"), 0.4),
      _ranked(_host("h4", "web01"), 0.4),
  ]
  expected = ranking.group_hosts(ranked, _settings())

  results = [
      ranking.group_hosts(list(permutation), _settings())
      for permutation in itertools.permutations(ranked)
  ]

  assert all(result == expected for result in results)


def test_group_hosts_without_hosts_returns_no_groups():
  assert not ranking.group_hosts([], _settings())


def test_group_hosts_folds_the_output_of_rank_hosts():
  affected = [
      (_host("h1", "worker-a", security_group="workers-sg"), ["100"]),
      (_host("h2", "worker-b", security_group="workers-sg"), ["100"]),
      (_host("h3", "worker-c", security_group="workers-sg"), ["100"]),
      (
          _host(
              "h4",
              "bastion",
              criticality=5,
              internet_facing=True,
              security_group="edge-sg",
          ),
          ["100"],
      ),
  ]
  ranked, _ = ranking.rank_hosts(
      affected, [_match("100", severity=4)], _settings()
  )

  groups = ranking.group_hosts(ranked, _settings())

  assert [(g.label, g.count, g.internet_facing_count) for g in groups] == [
      ("edge-sg", 1, 1),
      ("workers-sg", 3, 0),
  ]
  assert [g.priority for g in groups] == pytest.approx([4 / 3, 0.4])
