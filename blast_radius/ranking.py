"""Deterministic priority ranking and grouping of affected hosts.

Which hosts are affected is settled by a SQL join before this module runs.
What is left is to say which of them to fix first, and to fold hundreds of
like hosts into a few lines of work. Both are arithmetic over fields of the
scanner exports, so the same input always gives the same order, and no
language model takes part:

    threat   = w_sev * severity + w_epss * epss_percentile
               + w_kev * known_exploited
    exposure = (internet_facing_multiplier if internet_facing else 1)
               * (criticality / baseline_criticality)
    priority = threat * exposure

``threat`` describes a QID and ``exposure`` a host. A host with several
matched QIDs takes the threat of the worst of them, its driving QID. Every
weight and constant comes from ``config.Settings``, and every factor is
returned beside the score, because the data holds no ground truth for
priority: the ordering cannot be evaluated, so it has to be explainable. The
export's own risk score is reported next to the factors for comparison and
is never an input.

Hosts that are not running are set aside unranked. The running ones are
then grouped by security group, falling back to the cluster and then to the
shape of the host name, so that a few hundred interchangeable workers read
as one unit of work.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math
import re

from blast_radius import config
from blast_radius import models

_CVSS_MAX = 10.0
_QUALYS_SEVERITY_MAX = 5.0

# Stands in for the part of a host name that differs between like hosts.
_PLACEHOLDER = "#"
# The address that cloud DNS embeds in a name ("ip-10-1-2-3.eu-west-1..."):
# four groups of up to three digits joined by dashes or dots. The lookarounds
# keep it from matching part of a longer run of digits.
_IPV4_IN_NAME = re.compile(
    r"(?<![0-9])[0-9]{1,3}(?:[-.][0-9]{1,3}){3}(?![0-9])"
)
# A whole token of eight or more hexadecimal characters: an instance id or a
# generated suffix. Eight is the shortest such id in common use, and ordinary
# words of that length are never spelt with the letters a-f alone.
_HEX_ID = re.compile(r"(?<![0-9A-Za-z])[0-9A-Fa-f]{8,}(?![0-9A-Za-z])")
_DIGIT_RUN = re.compile(r"[0-9]+")


@dataclasses.dataclass(frozen=True)
class QidThreat:
  """How dangerous one matched QID is, with the inputs to that score.

  Attributes:
    qid: The check.
    severity: CVSS / 10 where a CVSS score exists, else Qualys severity / 5;
      0-1.
    epss_percentile: Highest EPSS percentile among the QID's CVEs, 0-1.
    known_exploited: Whether any of the QID's CVEs is in KEV.
    threat: Weighted blend of the three fields above.
    cogent_risk_score: Highest export risk score among the QID's CVEs, or
      None when none has one. Reported for comparison; never an input to
      ``threat``.
  """

  qid: str
  severity: float
  epss_percentile: float
  known_exploited: bool
  threat: float
  cogent_risk_score: float | None = None


def _clamp_unit(value: float) -> float:
  """Returns ``value`` limited to the range 0-1, counting NaN as 0.

  The exports should never hold a score outside its scale. If one did, it
  must not be able to lift a host above every honest score, and a NaN,
  which compares false with everything, must not be able to make the sort
  order depend on the order of the input.
  """
  if math.isnan(value):
    return 0.0
  return min(1.0, max(0.0, value))


def _severity(match: models.QidMatch) -> float:
  """Returns the severity of a matched QID on a 0-1 scale.

  With a Qualys severity of 3:

    CVEs scored 9.8, 7.5 and unscored   ->  9.8 / 10 = 0.98
    CVEs all unscored, or no CVEs       ->  3 / 5    = 0.6

  Args:
    match: The QID, with its Qualys severity and its CVEs.
  """
  scores = [
      _clamp_unit(cve.cvss / _CVSS_MAX)
      for cve in match.cves
      if cve.cvss is not None
  ]
  if scores:
    return max(scores)
  # CVSS is missing for more than half of the CVEs in the reference dataset,
  # and its most widespread explained finding has no CVE at all, so the
  # scanner's own severity has to stand in.
  if match.severity is not None:
    return _clamp_unit(match.severity / _QUALYS_SEVERITY_MAX)
  return 0.0


def _epss_percentile(cves: Sequence[models.CveSummary]) -> float:
  """Returns the highest EPSS percentile among ``cves``, or 0 without one."""
  percentiles = [
      _clamp_unit(cve.epss_percentile)
      for cve in cves
      if cve.epss_percentile is not None
  ]
  return max(percentiles, default=0.0)


def _risk_score(cves: Sequence[models.CveSummary]) -> float | None:
  """Returns the highest export risk score among ``cves``, if any has one."""
  scores = [
      cve.cogent_risk_score for cve in cves if cve.cogent_risk_score is not None
  ]
  return max(scores, default=None)


def _threat(
    severity: float,
    epss_percentile: float,
    known_exploited: bool,
    settings: config.Settings,
) -> float:
  """Returns the weighted blend of the three threat signals.

  With the default weights of 0.5, 0.3 and 0.2, a QID of severity 0.8 and
  EPSS percentile 0.5 that is known to be exploited scores

    0.5 * 0.8 + 0.3 * 0.5 + 0.2 * 1 = 0.75

  The weights are used exactly as configured. If they do not sum to 1 the
  result leaves the 0-1 scale, but the order it produces is still well
  defined, so they are not renormalised: the factors shown in a response
  must be reproducible from the configuration the operator wrote.

  Args:
    severity: Severity on a 0-1 scale.
    epss_percentile: EPSS percentile, 0-1.
    known_exploited: Whether the QID has a CVE in KEV.
    settings: Supplies the three weights.
  """
  return (
      settings.weight_severity * severity
      + settings.weight_epss * epss_percentile
      + settings.weight_known_exploited * float(known_exploited)
  )


def qid_threat(match: models.QidMatch, settings: config.Settings) -> QidThreat:
  """Returns the threat score of one matched QID, with its inputs.

  A QID bundles up to hundreds of CVEs and is judged by the worst of them:
  each signal is the maximum over the QID's CVEs, taken separately. Severity
  is the highest CVSS score over 10. Only a QID none of whose CVEs has a CVSS
  score, or that has no CVEs, falls back to its own Qualys severity over 5,
  and to 0 when that is missing too. Inputs outside their scale are clamped.

  Args:
    match: The QID and every CVE the scanner maps to it.
    settings: Supplies the three threat weights.
  """
  severity = _severity(match)
  epss_percentile = _epss_percentile(match.cves)
  known_exploited = any(cve.known_exploited for cve in match.cves)
  return QidThreat(
      qid=match.qid,
      severity=severity,
      epss_percentile=epss_percentile,
      known_exploited=known_exploited,
      threat=_threat(severity, epss_percentile, known_exploited, settings),
      cogent_risk_score=_risk_score(match.cves),
  )


def exposure(host: models.Host, settings: config.Settings) -> float:
  """Returns the multiplier that ``host`` applies to a threat.

  With the default multiplier of 2 and baseline criticality of 3:

    internet-facing, criticality 5   ->  2 * (5 / 3) = 3.33
    internal, criticality 3          ->  1 * (3 / 3) = 1.0

  Args:
    host: The host.
    settings: Supplies the multiplier and the baseline criticality.
  """
  reach = settings.internet_facing_multiplier if host.internet_facing else 1.0
  return reach * (host.criticality / settings.baseline_criticality)


def _priority_order(ranked_host: models.RankedHost) -> tuple[float, str, str]:
  """Returns the sort key that puts the most urgent host first.

  Name and then id break ties, so that hosts of equal priority, which is
  most of any large group, come out in the same order on every run.
  """
  return (-ranked_host.priority, ranked_host.host.name, ranked_host.host.id)


def _rank_host(
    host: models.Host,
    qids: Sequence[str],
    threats: Mapping[str, QidThreat],
    settings: config.Settings,
) -> models.RankedHost:
  """Returns ``host`` with its priority and every factor behind it.

  Args:
    host: A running host.
    qids: The matched QIDs detected on it; not empty.
    threats: The threat of every matched QID, by QID.
    settings: Supplies the exposure constants.
  """
  # The highest threat drives. On a tie the QID that sorts first does, so
  # that the choice never depends on the order the detections arrived in.
  driving = min(
      (threats[qid] for qid in qids),
      key=lambda candidate: (-candidate.threat, candidate.qid),
  )
  host_exposure = exposure(host, settings)
  return models.RankedHost(
      host=host,
      qids=list(qids),
      priority=driving.threat * host_exposure,
      factors=models.PriorityFactors(
          severity=driving.severity,
          epss_percentile=driving.epss_percentile,
          known_exploited=driving.known_exploited,
          threat=driving.threat,
          internet_facing=host.internet_facing,
          criticality=host.criticality,
          exposure=host_exposure,
          driving_qid=driving.qid,
          cogent_risk_score=driving.cogent_risk_score,
      ),
  )


def rank_hosts(
    affected: Sequence[tuple[models.Host, Sequence[str]]],
    matches: Sequence[models.QidMatch],
    settings: config.Settings,
) -> tuple[list[models.RankedHost], list[models.Host]]:
  """Ranks the running affected hosts and sets the others aside.

  A host's threat is the highest threat among the matched QIDs detected on
  it. That QID is reported as the host's driving QID; on a tie it is the one
  that sorts first as a string.

  Args:
    affected: Each affected host with the matched QIDs detected on it, as
      the store's blast-radius join returns them.
    matches: The matched QIDs. Every QID in ``affected`` must be among them.
    settings: Supplies the threat weights and the exposure constants.

  Returns:
    The running hosts, by priority descending, then name, then id, each
    with its QIDs in the order given and every factor filled in; and the
    hosts that are not running, by name, then id. Acting on a terminated
    instance is wasted work, so those are not ranked. They are returned so
    that the analyst can still see them.

  Raises:
    ValueError: If ``affected`` holds a QID that is not in ``matches``, or a
      host with no QIDs. Either is a programming error in the caller, not a
      property of the data.
  """
  threats = {match.qid: qid_threat(match, settings) for match in matches}
  for host, qids in affected:
    if not qids:
      raise ValueError(f"host {host.id} is listed as affected by no QID")
    unmatched = ", ".join(sorted(set(qids) - threats.keys()))
    if unmatched:
      raise ValueError(
          f"host {host.id} has QIDs that are not among the matches:"
          f" {unmatched}"
      )

  ranked = [
      _rank_host(host, qids, threats, settings)
      for host, qids in affected
      if host.is_running
  ]
  inactive = [host for host, _ in affected if not host.is_running]
  return (
      sorted(ranked, key=_priority_order),
      sorted(inactive, key=lambda host: (host.name, host.id)),
  )


def name_pattern(name: str) -> str:
  """Returns ``name`` with the parts that vary between like hosts as ``#``.

  Embedded IPv4 addresses, hex-like id tokens and any remaining runs of
  digits are each replaced, in that order:

    ip-10-1-2-3.eu-west-1.compute.internal   ->  ip-#.eu-west-#.compute.internal
    build-agent-0a1b2c3d                     ->  build-agent-#
    web01                                    ->  web#

  Hosts launched from one template differ only in those parts, so an equal
  pattern is the best evidence of "the same kind of host" that a name holds.

  Args:
    name: A host name.
  """
  pattern = _IPV4_IN_NAME.sub(_PLACEHOLDER, name)
  pattern = _HEX_ID.sub(_PLACEHOLDER, pattern)
  return _DIGIT_RUN.sub(_PLACEHOLDER, pattern)


def _group_key(host: models.Host) -> tuple[str, str]:
  """Returns the grouping key of ``host`` and the label shown for it.

  The first of these that the host has: its security group, its cluster,
  the pattern of its name. An empty string counts as missing. A security
  group field that lists several groups is used whole: hosts are one unit
  of work when they share the entire set, not one member of it.

  Args:
    host: The host to place.
  """
  if host.security_group:
    return f"sg:{host.security_group}", host.security_group
  if host.cluster:
    return f"cluster:{host.cluster}", host.cluster
  pattern = name_pattern(host.name)
  return f"name:{pattern}", pattern


def _build_group(
    key: str,
    label: str,
    members: Sequence[models.RankedHost],
    example_count: int,
) -> models.HostGroup:
  """Returns the group made of ``members``.

  Args:
    key: The grouping key the members share.
    label: What to show for the group.
    members: The group's hosts in priority order; not empty.
    example_count: How many host names to list as examples.
  """
  # Hosts launched from one template often share a name as well, and the
  # same name four times over would tell the reader nothing. ``fromkeys``
  # drops the repeats and keeps the order.
  distinct_names = list(dict.fromkeys(member.host.name for member in members))
  return models.HostGroup(
      key=key,
      label=label,
      count=len(members),
      priority=members[0].priority,
      internet_facing_count=sum(
          1 for member in members if member.host.internet_facing
      ),
      example_hosts=distinct_names[:example_count],
      host_ids=[member.host.id for member in members],
  )


def group_hosts(
    ranked: Sequence[models.RankedHost], settings: config.Settings
) -> list[models.HostGroup]:
  """Folds ranked hosts that can be treated as one unit of work.

  Hosts behind one security group were launched for the same job and are
  patched together, so that is the first choice of key (``sg:<name>``).
  A host without one falls back to its cluster (``cluster:<name>``) and
  then to the pattern of its name (``name:<pattern>``, see
  ``name_pattern``). A group's label is its key without the prefix.

  Args:
    ranked: The ranked hosts, in any order.
    settings: Supplies how many example host names each group lists.

  Returns:
    The groups, by priority descending, then host count descending, then
    label, then key, since a security group and a cluster may share a name.
    Each takes the priority of its most urgent host, and lists the ids of
    all its hosts and its first few distinct host names, most urgent first.
  """
  by_key: dict[tuple[str, str], list[models.RankedHost]] = {}
  for ranked_host in sorted(ranked, key=_priority_order):
    by_key.setdefault(_group_key(ranked_host.host), []).append(ranked_host)
  groups = [
      _build_group(key, label, members, settings.group_example_count)
      for (key, label), members in by_key.items()
  ]
  return sorted(
      groups,
      key=lambda group: (-group.priority, -group.count, group.label, group.key),
  )
