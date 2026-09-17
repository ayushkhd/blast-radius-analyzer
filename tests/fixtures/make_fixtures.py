"""Generates the synthetic scanner exports that the tests run on.

Run it from the repository root and commit what it writes:

  uv run python tests/fixtures/make_fixtures.py

It writes ``assets.json`` and ``vulns.json`` beside itself, in the shape of
the two Qualys exports the product reads: the same keys, the same nesting
and the same quirks (numbers and booleans as strings, a full copy of the
host record inside every vulns row, rows with no CVE enrichment, references
with and without tags). Everything in them is invented. Addresses come from
the documentation range of RFC 5737 and the private range of RFC 1918,
domains end in ``.example``, and CVE ids use the year 2099.

The script is deterministic, with no randomness and no clock, so running it
again reproduces the committed files byte for byte, which a test checks.

The environment it describes:

* Two internet-facing hosts of criticality 5, a bastion and a NAT gateway.
  The bastion carries the "Internet Facing Assets" tag; the gateway is
  internet-facing by its public IP alone.
* Five Kubernetes workers in two security groups. Two share one name, as
  autoscaled nodes do; one names its cluster only through a
  ``kubernetes.io/cluster/<name>`` tag key; one has the same QID detected
  twice; one is terminated.
* A web server, a message-queue host, a build agent, and a runner on which
  nothing was detected.
* Six explained QIDs, described where ``_FINDINGS`` defines them, and two
  unexplained QIDs that sit on nearly every host and appear nowhere in
  ``vulns.json``.
"""

import copy
import dataclasses
import json
import pathlib
from typing import Any

QID_OPENSSH = "710001"
QID_KERNEL = "710002"
QID_PROXY = "710003"
QID_WEB_SERVER = "710004"
QID_BROKER = "710005"
QID_AGENT = "710006"
UNEXPLAINED_QIDS = ("710901", "710902")

_REGION = "us-west-2"
_FIRST_FOUND = "2099-02-01T06:00:00Z"
_LAST_FOUND = "2099-03-01T06:00:00Z"
_FIRST_DETECTION_ID = 51000000001

_NVD = "nvd@nist.example"
_CNA = "cna@vendor.example"

_TAG_CONNECTOR = ("70000001", "Fixture Cloud Connector")
_TAG_WORKERS = ("70000002", "fixture-k8s-workers")
_TAG_INTERNET_FACING = ("70000003", "Internet Facing Assets")

# The position of a group in this tuple gives its ``groupId``.
_SECURITY_GROUPS = (
    "fx-bastion-sg",
    "fx-gateway-sg",
    "fx-k8s-workers-a-sg",
    "fx-k8s-workers-b-sg",
    "fx-web-sg",
    "fx-data-sg",
    "fx-build-sg",
)

# What the scanner recognised on a port: (serviceName, serviceId). A port
# that is not listed here was found open with nothing recognised behind it.
_SERVICES = {
    (22, "TCP"): ("ssh", "2001"),
    (80, "TCP"): ("http", "2002"),
    (111, "TCP"): ("rpc", "2003"),
    (111, "UDP"): ("rpc_udp", "2004"),
    (443, "TCP"): ("http", "2002"),
    (8080, "TCP"): ("proxy_http", "2005"),
    (15672, "TCP"): ("http", "2002"),
}

_KERNEL_BOILERPLATE = (
    "In the Linux kernel, the following vulnerability has been resolved:\n\n"
)

_FXNET_DESCRIPTION = """\
fxnet: fix use-after-free of the RX ring in fxnet_ring_teardown()

When the MTU of an fxnet interface is changed while a reset is pending,
fxnet_change_mtu() frees the RX descriptor ring and allocates a new one,
but the reset worker that was queued earlier still holds a pointer to the
old ring. The worker then walks the freed ring in fxnet_ring_teardown()
and reads the buffer list of every descriptor, which is a use-after-free.
A local user who is allowed to configure the interface can trigger the
race by changing the MTU in a loop while the link flaps.

KASAN reports the problem as follows:

BUG: KASAN: slab-use-after-free in fxnet_ring_teardown+0x1a4/0x2b0 [fxnet]
Read of size 8 at addr ffff8881a2b3c4d8 by task kworker/u16:3/2114
CPU: 5 PID: 2114 Comm: kworker/u16:3 Tainted: G W 6.8.0-fx #1
Workqueue: fxnet_wq fxnet_reset_task [fxnet]
RIP: 0010:fxnet_ring_teardown+0x1a4/0x2b0 [fxnet]
Code: 48 8b 45 08 48 85 c0 74 1e 48 8b 38 e8 5b 2c ff ff 48 8b 45 08
 <48> 8b 40 18 48 85 c0 75 e6 49 8d 7c 24 08 e8 37 5e d0 de 49 8b 1c
RSP: 0018:ffffc90004a2fd10 EFLAGS: 00010246
RAX: 0000000000000000 RBX: ffff8881a2b3c4c0 RCX: dffffc0000000000
RDX: 1ffff11034567899 RSI: 0000000000000008 RDI: ffff8881a2b3c4d8
RBP: ffffc90004a2fd58 R08: 0000000000000001 R09: ffffed1034567899
R10: ffff8881a2b3c4df R11: 0000000000000000 R12: ffff88810c5e9000
R13: ffff88810c5e9a80 R14: 0000000000000040 R15: ffff8881a2b3c400
FS: 0000000000000000(0000) GS:ffff8881f6d40000(0000)
CS: 0010 DS: 0000 ES: 0000 CR0: 0000000080050033
CR2: ffff8881a2b3c4d8 CR3: 000000010e1a6002 CR4: 00000000003706e0
Call Trace:
 <TASK>
 dump_stack_lvl+0x5d/0x80
 print_report+0x174/0x505
 kasan_report+0xd0/0x150
 ? fxnet_ring_teardown+0x1a4/0x2b0 [fxnet]
 fxnet_ring_teardown+0x1a4/0x2b0 [fxnet]
 fxnet_down+0x2e1/0x4a0 [fxnet]
 fxnet_reset_task+0x9c/0x1f0 [fxnet]
 process_one_work+0x5e2/0xff0
 worker_thread+0x8c6/0x1290
 kthread+0x2d3/0x3a0
 ret_from_fork+0x31/0x70
 ret_from_fork_asm+0x1a/0x30
 </TASK>

Allocated by task 1870:
 kasan_save_stack+0x33/0x60
 kasan_save_track+0x14/0x30
 __kasan_kmalloc+0xaa/0xb0
 fxnet_ring_alloc+0x7b/0x310 [fxnet]
 fxnet_open+0x1c9/0x720 [fxnet]
 __dev_open+0x28b/0x460
 __dev_change_flags+0x4a5/0x6e0

Freed by task 2391:
 kasan_save_stack+0x33/0x60
 kasan_save_track+0x14/0x30
 kasan_save_free_info+0x3b/0x60
 kfree+0x11d/0x3a0
 fxnet_ring_free+0x95/0x140 [fxnet]
 fxnet_change_mtu+0x21a/0x3c0 [fxnet]
 dev_set_mtu_ext+0x3ab/0x610

Fix this by taking the ring lock in fxnet_change_mtu() before the old ring
is freed, and by cancelling the pending reset work and waiting for it to
finish before the ring pointer is replaced. The reset worker now looks the
ring up again under the same lock instead of using the pointer it cached
when it was queued, so it can no longer see a ring that has been freed."""

_QUARTZFS_DESCRIPTION = """\
quartzfs: reject directory entries that run past the end of the block

quartzfs_readdir() trusts the name length stored in an on-disk directory
entry. A crafted image can set the length so that the name extends beyond
the block that was read, and the copy to user space then reads adjacent
kernel memory. Validate the entry against the block size in
quartzfs_check_dirent() and return -EFSCORRUPTED when it does not fit."""

_TIDAL_DESCRIPTION = """\
tidal: fix NULL pointer dereference in tidal_queue_flush()

tidal_queue_flush() can run after tidal_dev_remove() has cleared
dev->queue, because the flush timer is only stopped once the queue has
been released. Stop the timer with timer_shutdown_sync() before the queue
is released, and check the pointer under the device lock."""

_LUMEN_DESCRIPTION = """\
lumen: avoid a deadlock between lumen_irq_thread() and system suspend

lumen_irq_thread() takes the panel mutex and then waits for the runtime PM
reference, while lumen_suspend() holds the runtime PM lock and waits for
the panel mutex. If an interrupt arrives while the system is suspending,
both paths block for ever and the machine hangs. Take the runtime PM
reference before the panel mutex in the interrupt thread, which is the
order every other path already uses."""


@dataclasses.dataclass(frozen=True)
class _Reference:
  """One NVD reference. ``tags`` of None leaves the key out, as NVD does."""

  url: str
  tags: tuple[str, ...] | None = None


@dataclasses.dataclass(frozen=True)
class _CveSpec:
  """The enrichment the vulns export carries for one CVE.

  Attributes:
    cve_id: The identifier, always in the year 2099.
    title: Short title.
    description: NVD-style plain-text description.
    cvss: CVSS base score, or None where NVD has not scored the CVE.
    attack_vector: Lower-case attack vector, or "" where there is no CVSS.
    epss: EPSS probability.
    epss_percentile: EPSS percentile.
    risk_score: The export's own 0-10 risk score.
    published: Publication timestamp in the export's format.
    weaknesses: ``(source, value)`` pairs, or None to leave the key out.
    references: NVD references.
    known_exploit: The KEV catalogue entry, or None when not in KEV.
    how_to_fix: The export's fix text; "N/A" on almost every real row.
  """

  cve_id: str
  title: str
  description: str
  cvss: float | None
  attack_vector: str
  epss: float
  epss_percentile: float
  risk_score: float
  published: str
  weaknesses: tuple[tuple[str, str], ...] | None
  references: tuple[_Reference, ...]
  known_exploit: dict[str, str] | None = None
  how_to_fix: str = "N/A"


@dataclasses.dataclass(frozen=True)
class _FindingSpec:
  """One explained QID: its QID-level fields and the CVEs it bundles.

  Attributes:
    qid: The check id.
    category: Qualys category.
    severity: Qualys severity, 1-5.
    pci_flag: 1 when the finding fails PCI compliance.
    diagnosis: The check's write-up, an HTML fragment.
    cves: The CVEs with enrichment.
    unenriched_cves: How many more CVEs the diagnosis names that the export
      has no enrichment for. Each adds a row with QID-level keys only.
  """

  qid: str
  category: str
  severity: int
  pci_flag: int
  diagnosis: str
  cves: tuple[_CveSpec, ...] = ()
  unenriched_cves: int = 0


@dataclasses.dataclass(frozen=True)
class _HostSpec:
  """What distinguishes one host; everything else is derived or constant.

  Attributes:
    number: Small integer that every generated id of the host is built from.
    name: Display name, also the EC2 ``Name`` tag.
    security_group: Name of the security group, one of ``_SECURITY_GROUPS``.
    qids: QIDs detected on the host, in the order the scanner lists them.
    ports: Open ``(port, protocol)`` pairs.
    public_ip: Public address, for the two internet-facing hosts.
    internet_facing_tag: Whether the host carries the scanner's
      "Internet Facing Assets" tag.
    criticality: The scanner's 1-5 criticality.
    state: EC2 instance state.
    ec2_tags: EC2 tags other than ``Name``.
    docker: Whether the scanner found a container runtime.
    worker: Whether the host is a Kubernetes worker.
  """

  number: int
  name: str
  security_group: str
  qids: tuple[str, ...]
  ports: tuple[tuple[int, str], ...]
  public_ip: str | None = None
  internet_facing_tag: bool = False
  criticality: int = 3
  state: str = "RUNNING"
  ec2_tags: tuple[tuple[str, str], ...] = ()
  docker: bool = False
  worker: bool = False


def _kernel_cve(
    number: int,
    title: str,
    description: str,
    *,
    cvss: float | None,
    epss_percentile: float,
    risk_score: float,
    weaknesses: tuple[tuple[str, str], ...] | None,
    patch_tagged: bool,
) -> _CveSpec:
  """Returns a kernel CVE: local, low EPSS, one reference to the commit.

  Args:
    number: The last part of the CVE id.
    title: Short title.
    description: The commit message, without the sentence that opens every
      kernel CVE, which is added here.
    cvss: CVSS base score, or None for a CVE that NVD has not analysed,
      which then has no attack vector either.
    epss_percentile: EPSS percentile.
    risk_score: The export's own risk score.
    weaknesses: ``(source, value)`` pairs, or None to leave the key out.
    patch_tagged: Whether NVD has tagged the commit reference "Patch".
  """
  commit = f"https://git.example.org/linux/c/{number}a1b2c3d4e5f6"
  return _CveSpec(
      cve_id=f"CVE-2099-{number}",
      title=title,
      description=_KERNEL_BOILERPLATE + description,
      cvss=cvss,
      attack_vector="" if cvss is None else "local",
      epss=0.00045,
      epss_percentile=epss_percentile,
      risk_score=risk_score,
      published="2099-02-18T10:15:00.000000",
      weaknesses=weaknesses,
      references=(_Reference(commit, ("Patch",) if patch_tagged else None),),
  )


_FINDINGS: tuple[_FindingSpec, ...] = (
    # 1. No CVE at all. The diagnosis opens with a sentence about the
    # product, not the weakness, and states the affected versions.
    _FindingSpec(
        qid=QID_OPENSSH,
        category="General remote services",
        severity=4,
        pci_flag=1,
        diagnosis=(
            "OpenSSH is a suite of tools for encrypted remote login and file"
            " transfer over the SSH protocol.<P>\n\n"
            "OpenSSH may allow an authentication bypass on hardware prone to"
            " memory bit flips, because the server keeps the authenticated"
            " state of a session in a single integer that does not resist"
            " the flip of one bit.<P>\n\n"
            "Affected Versions:<BR>OpenSSH up to version 9.6<P>\n\n"
            "QID Detection Logic:<BR>\n"
            "This unauthenticated check reads the version that the SSH"
            " service reports in its banner.<P>"
        ),
    ),
    # 2. A distribution update that bundles four kernel CVEs. CVE-2099-1001
    # is long and has a call trace in the middle; CVE-2099-1002 has no CVSS,
    # no weaknesses and an untagged reference. They are out of id order, as
    # the rows of a real export are.
    _FindingSpec(
        qid=QID_KERNEL,
        category="Ubuntu",
        severity=4,
        pci_flag=1,
        diagnosis=(
            "Ubuntu has released a security update for linux to fix the"
            " vulnerabilities.<BR><BR><P>QID Detection Logic"
            " (Authenticated):<BR>The check lists the installed packages"
            " with the system package manager, such as &quot;dpkg&quot;,"
            " and compares their versions with the fixed versions in the"
            " vendor advisory.<BR>"
        ),
        cves=(
            _kernel_cve(
                1003,
                "Linux Kernel tidal NULL Pointer Dereference in Queue Flush",
                _TIDAL_DESCRIPTION,
                cvss=5.5,
                epss_percentile=0.15,
                risk_score=3.2,
                weaknesses=((_NVD, "CWE-476"),),
                patch_tagged=True,
            ),
            _kernel_cve(
                1001,
                "Linux Kernel fxnet Use-After-Free in RX Ring Teardown",
                _FXNET_DESCRIPTION,
                cvss=7.8,
                epss_percentile=0.21,
                risk_score=4.4,
                weaknesses=((_NVD, "CWE-416"),),
                patch_tagged=True,
            ),
            _kernel_cve(
                1004,
                "Linux Kernel lumen Deadlock Between Interrupt Thread and"
                " Suspend",
                _LUMEN_DESCRIPTION,
                cvss=4.7,
                epss_percentile=0.11,
                risk_score=2.8,
                weaknesses=((_NVD, "NVD-CWE-noinfo"), (_CNA, "CWE-667")),
                patch_tagged=True,
            ),
            _kernel_cve(
                1002,
                "Linux Kernel quartzfs Out-of-Bounds Read in Directory Listing",
                _QUARTZFS_DESCRIPTION,
                cvss=None,
                epss_percentile=0.08,
                risk_score=1.9,
                weaknesses=None,
                patch_tagged=False,
            ),
        ),
    ),
    # 3. One network-reachable CVE with a patch and a vendor advisory.
    _FindingSpec(
        qid=QID_PROXY,
        category="CGI",
        severity=4,
        pci_flag=1,
        diagnosis=(
            "Trellis proxy is an open-source reverse proxy and load balancer"
            " for HTTP services. Before it routes a request, Trellis proxy"
            " adds headers such as X-Forwarded-Host and X-Forwarded-Port,"
            " which the application behind it trusts. A client can list"
            " those headers in the Connection header, so that they are"
            " treated as hop-by-hop headers and removed again before the"
            " request reaches the application.\n<P>\n"
            "Affected Versions <BR>\n"
            "Trellis proxy before 2.8.4, and 3.0.0 before 3.1.2<P>\n\n"
            "QID Detection Logic (Unauthenticated)<BR>\n"
            "The check requests /api/version and compares the version in"
            " the response."
        ),
        cves=(
            _CveSpec(
                cve_id="CVE-2099-2001",
                title="Trellis Proxy Forwarded Header Removal Vulnerability",
                description=(
                    "Trellis proxy before 2.8.4 and 3.x before 3.1.2 lets a"
                    " remote client remove the X-Forwarded-Host and"
                    " X-Forwarded-Port headers that the proxy adds, by naming"
                    " them in the HTTP/1.1 Connection header. An application"
                    " that trusts those headers can be made to build links"
                    " to, or apply access rules for, the wrong host."
                ),
                cvss=7.5,
                attack_vector="network",
                epss=0.00212,
                epss_percentile=0.59,
                risk_score=6.4,
                published="2099-03-05T14:20:00.000000",
                weaknesses=((_CNA, "CWE-345"), (_NVD, "CWE-348")),
                references=(
                    _Reference(
                        "https://git.example.org/trellis/proxy/commit/5f2d9c1",
                        ("Patch",),
                    ),
                    _Reference(
                        "https://trellis.example/security/TSA-2099-003",
                        ("Vendor Advisory",),
                    ),
                    _Reference("https://trellis.example/releases/3.1.2"),
                ),
            ),
        ),
    ),
    # 4. Two CVEs with enrichment, and a third that the diagnosis names but
    # the export could not enrich.
    _FindingSpec(
        qid=QID_WEB_SERVER,
        category="CGI",
        severity=3,
        pci_flag=1,
        diagnosis=(
            "Marlin HTTP Server is an open-source web server for static and"
            " proxied content.<P>\n\n"
            "CVE-2099-3001 - A request whose chunked body is followed by"
            " extra data is forwarded by mod_relay without being normalised,"
            " which lets a remote attacker smuggle a second request to the"
            " backend.<BR>\n"
            "CVE-2099-3002 - A template that includes itself can make"
            " mod_template read past the end of its expansion buffer and"
            " crash the worker process.<BR>\n"
            "CVE-2099-3003 - The status page discloses the paths of"
            " configuration files to unauthenticated clients.<P>\n\n"
            "Affected Versions:<BR>\n"
            "Marlin HTTP Server versions prior to 3.2.9<P>\n\n"
            "QID Detection Logic:(Unauthenticated)<BR>\n"
            "The check reads the Server response header to find the version"
            " of Marlin HTTP Server.<P>"
        ),
        cves=(
            _CveSpec(
                cve_id="CVE-2099-3001",
                title="Marlin HTTP Server mod_relay HTTP Request Smuggling",
                description=(
                    "Marlin HTTP Server before 3.2.9 does not normalise a"
                    " chunked request body that is followed by trailing data"
                    " when mod_relay forwards it. A remote attacker can use"
                    " this to smuggle a second request to the backend server"
                    " and bypass access rules enforced by the front end."
                ),
                cvss=8.2,
                attack_vector="network",
                epss=0.00391,
                epss_percentile=0.73,
                risk_score=6.9,
                published="2099-01-20T09:00:00.000000",
                weaknesses=((_NVD, "CWE-444"),),
                references=(
                    _Reference(
                        "https://lists.example.org/marlin-announce/2099/0007",
                        ("Mailing List", "Third Party Advisory"),
                    ),
                    _Reference(
                        "https://git.example.org/marlin/httpd/commit/8c41e07",
                        ("Patch",),
                    ),
                    # The same advisory again under the other advisory tag.
                    _Reference(
                        "https://lists.example.org/marlin-announce/2099/0007",
                        ("Vendor Advisory",),
                    ),
                ),
            ),
            _CveSpec(
                cve_id="CVE-2099-3002",
                title="Marlin HTTP Server mod_template Out-of-Bounds Read",
                description=(
                    "mod_template in Marlin HTTP Server before 3.2.9 reads"
                    " beyond the end of its expansion buffer when a template"
                    " includes itself, as shown by a file that contains"
                    " <include self>. A remote attacker who can upload"
                    " templates can crash the worker process."
                ),
                cvss=5.3,
                attack_vector="network",
                epss=0.00087,
                epss_percentile=0.37,
                risk_score=3.8,
                published="2099-01-20T09:05:00.000000",
                weaknesses=(
                    (_NVD, "CWE-125"),
                    (_NVD, "NVD-CWE-Other"),
                    (_CNA, "CWE-125"),
                    (_CNA, "CWE-1284"),
                ),
                references=(
                    _Reference("https://marlin.example/changes/3.2.9"),
                    _Reference(
                        "https://marlin.example/security/MSA-2099-02",
                        ("Release Notes",),
                    ),
                ),
            ),
        ),
        unenriched_cves=1,
    ),
    # 5. A known-exploited CVE, with a KEV required action and a vendor fix
    # that differ, so that a test can tell the two apart.
    _FindingSpec(
        qid=QID_BROKER,
        category="General remote services",
        severity=5,
        pci_flag=1,
        diagnosis=(
            "Ferrous broker is a message broker that speaks AMQP and"
            " MQTT.<P>\n\n"
            "The management listener deserialises the body of a cluster join"
            " request before it checks the sender's credentials, so a remote"
            " attacker who can reach the listener can run arbitrary code as"
            " the broker's service account.<P>\n\n"
            "Affected Versions:<BR>\n"
            "Ferrous broker 4.0.0 to 4.1.1<P>\n\n"
            "QID Detection Logic (Unauthenticated):<BR>\n"
            "The check connects to the management listener on TCP port 15672"
            " and reads the version from its greeting.<P>"
        ),
        cves=(
            _CveSpec(
                cve_id="CVE-2099-4001",
                title=(
                    "Ferrous Broker Management Listener Remote Code"
                    " Execution"
                ),
                description=(
                    "Ferrous broker 4.0.0 through 4.1.1 deserialises cluster"
                    " join requests on its management listener before"
                    " authentication. A remote, unauthenticated attacker can"
                    " send a crafted join request and execute arbitrary code"
                    " with the privileges of the broker process. The issue is"
                    " fixed in 4.1.2."
                ),
                cvss=9.8,
                attack_vector="network",
                epss=0.91433,
                epss_percentile=0.99,
                risk_score=9.6,
                published="2099-03-28T16:45:00.000000",
                weaknesses=((_NVD, "CWE-502"),),
                references=(
                    _Reference(
                        "https://ferrous.example/security/FSA-2099-01",
                        ("Vendor Advisory",),
                    ),
                    _Reference(
                        "https://git.example.org/ferrous/broker/commit/d07be44",
                        ("Patch",),
                    ),
                    _Reference(
                        "https://kev.example.org/catalog/CVE-2099-4001",
                        ("US Government Resource",),
                    ),
                ),
                known_exploit={
                    "cveID": "CVE-2099-4001",
                    "vendorProject": "Ferrous",
                    "product": "Broker",
                    "dateAdded": "2099-04-02",
                    "vulnerabilityName": (
                        "Ferrous Broker Deserialization of Untrusted Data"
                        " Vulnerability"
                    ),
                    "shortDescription": (
                        "Ferrous Broker deserialises untrusted data on its"
                        " management listener, which allows remote code"
                        " execution without authentication."
                    ),
                    "requiredAction": (
                        "Apply the vendor's update, or stop exposing the"
                        " management listener until the update is applied."
                    ),
                    "dueDate": "2099-04-23",
                    "knownRansomwareCampaignUse": "Unknown",
                    "notes": "https://ferrous.example/security/FSA-2099-01",
                    "cwes": "CWE-502",
                },
                how_to_fix="Upgrade Ferrous broker to version 4.1.2 or later.",
            ),
        ),
    ),
    # 6. Low severity and local: no CVSS, no attack vector, no weaknesses.
    _FindingSpec(
        qid=QID_AGENT,
        category="Local",
        severity=2,
        pci_flag=0,
        diagnosis=(
            "The Pylon monitoring agent stores the results of its last run in"
            " a state file.<P>\n\n"
            "The state file is created with world-readable permissions, so"
            " any local user can read the process arguments and environment"
            " of the services that the agent inspected.<P>\n\n"
            "Affected Versions:<BR>\n"
            "Pylon agent before 1.4.0<P>\n\n"
            "QID Detection Logic (Authenticated):<BR>\n"
            "The check reads the mode of /var/lib/pylon/state.json.<P>"
        ),
        cves=(
            _CveSpec(
                cve_id="CVE-2099-5001",
                title="Pylon Agent World-Readable State File",
                description=(
                    "Pylon agent before 1.4.0 creates /var/lib/pylon/state.json"
                    " with mode 0644. A local user can read process arguments"
                    " and environment variables of other users' services from"
                    " the file."
                ),
                cvss=None,
                attack_vector="",
                epss=0.00043,
                epss_percentile=0.09,
                risk_score=1.2,
                published="2099-02-14T11:30:00.000000",
                weaknesses=None,
                references=(
                    _Reference("https://pylon.example/changelog#1.4.0"),
                ),
            ),
        ),
    ),
)

_SSH = (22, "TCP")
# Out of port order, as the scanner lists them.
_WORKER_PORTS = ((8080, "TCP"), _SSH, (111, "UDP"), (111, "TCP"), (80, "TCP"))
_DEV_CLUSTER_TAGS = (
    ("aws:eks:cluster-name", "fx-dev"),
    ("kubernetes.io/cluster/fx-dev", "owned"),
    ("role", "node"),
)
_BUILD_CLUSTER_TAGS = (
    ("aws:eks:cluster-name", "fx-build"),
    ("kubernetes.io/cluster/fx-build", "owned"),
    ("role", "node"),
)

_HOSTS: tuple[_HostSpec, ...] = (
    _HostSpec(
        number=1,
        name="fx-bastion",
        security_group="fx-bastion-sg",
        qids=(UNEXPLAINED_QIDS[0], QID_OPENSSH, UNEXPLAINED_QIDS[1]),
        ports=(_SSH,),
        public_ip="203.0.113.10",
        internet_facing_tag=True,
        criticality=5,
        ec2_tags=(("role", "bastion"),),
    ),
    # Internet-facing by its public IP alone: it lacks the scanner's tag.
    _HostSpec(
        number=2,
        name="fx-nat-gateway",
        security_group="fx-gateway-sg",
        qids=(*UNEXPLAINED_QIDS, QID_OPENSSH, QID_PROXY),
        ports=(_SSH, (443, "TCP")),
        public_ip="203.0.113.20",
        criticality=5,
        ec2_tags=(("role", "nat-gateway"),),
    ),
    _HostSpec(
        number=3,
        name="fx-k8s-worker-a",
        security_group="fx-k8s-workers-a-sg",
        qids=(*UNEXPLAINED_QIDS, QID_OPENSSH, QID_KERNEL, QID_PROXY),
        ports=_WORKER_PORTS,
        ec2_tags=_DEV_CLUSTER_TAGS,
        docker=True,
        worker=True,
    ),
    # Same name as the host before: autoscaled nodes share their Name tag.
    _HostSpec(
        number=4,
        name="fx-k8s-worker-a",
        security_group="fx-k8s-workers-a-sg",
        qids=(*UNEXPLAINED_QIDS, QID_OPENSSH, QID_KERNEL, QID_PROXY),
        ports=_WORKER_PORTS,
        ec2_tags=_DEV_CLUSTER_TAGS,
        docker=True,
        worker=True,
    ),
    # Names its cluster only in a tag key; it has no EKS tag.
    _HostSpec(
        number=5,
        name="fx-k8s-worker-b1",
        security_group="fx-k8s-workers-b-sg",
        qids=(*UNEXPLAINED_QIDS, QID_OPENSSH, QID_KERNEL, QID_PROXY),
        ports=_WORKER_PORTS,
        ec2_tags=(
            ("kubernetes.io/cluster/fx-build", "owned"),
            ("role", "node"),
        ),
        docker=True,
        worker=True,
    ),
    # The scanner reported the proxy finding twice on this host.
    _HostSpec(
        number=6,
        name="fx-k8s-worker-b2",
        security_group="fx-k8s-workers-b-sg",
        qids=(
            *UNEXPLAINED_QIDS,
            QID_OPENSSH,
            QID_KERNEL,
            QID_PROXY,
            QID_PROXY,
            QID_AGENT,
        ),
        ports=_WORKER_PORTS,
        ec2_tags=_BUILD_CLUSTER_TAGS,
        docker=True,
        worker=True,
    ),
    _HostSpec(
        number=7,
        name="fx-k8s-worker-b3",
        security_group="fx-k8s-workers-b-sg",
        qids=(UNEXPLAINED_QIDS[0], QID_OPENSSH, QID_KERNEL),
        ports=_WORKER_PORTS,
        state="TERMINATED",
        ec2_tags=_BUILD_CLUSTER_TAGS,
        docker=True,
        worker=True,
    ),
    _HostSpec(
        number=8,
        name="fx-web-01",
        security_group="fx-web-sg",
        qids=(*UNEXPLAINED_QIDS, QID_OPENSSH, QID_WEB_SERVER, QID_AGENT),
        ports=(_SSH, (80, "TCP"), (443, "TCP")),
    ),
    _HostSpec(
        number=9,
        name="fx-queue-01",
        security_group="fx-data-sg",
        qids=(*UNEXPLAINED_QIDS, QID_OPENSSH, QID_BROKER),
        ports=(_SSH, (5672, "TCP"), (15672, "TCP")),
    ),
    _HostSpec(
        number=10,
        name="fx-build-agent",
        security_group="fx-build-sg",
        qids=(*UNEXPLAINED_QIDS, QID_BROKER, QID_AGENT),
        ports=(_SSH, (15672, "TCP")),
        docker=True,
    ),
    # Known to the cloud connector but never scanned: no ports, no
    # detections and no scan time.
    _HostSpec(
        number=11,
        name="fx-idle-runner",
        security_group="fx-build-sg",
        qids=(),
        ports=(),
    ),
)


def _open_port(port: int, protocol: str) -> dict[str, str | None]:
  """Returns one ``openPorts`` entry; the export writes the port as text."""
  service_name, service_id = _SERVICES.get((port, protocol), (None, None))
  return {
      "port": str(port),
      "protocol": protocol,
      "serviceId": service_id,
      "serviceName": service_name,
  }


def _asset(host: _HostSpec, first_detection_id: int) -> dict[str, Any]:
  """Returns one record of the asset export.

  Args:
    host: The host to describe.
    first_detection_id: The ``hostInstanceVulnId`` of the host's first
      detection; the rest follow on, which keeps the ids globally unique.
  """
  n = host.number
  asset_id = str(900000000 + n)
  private_ip = f"10.20.{n}.10"
  private_dns = f"ip-10-20-{n}-10.{_REGION}.compute.internal"
  public_dns = None
  if host.public_ip:
    dashed = host.public_ip.replace(".", "-")
    public_dns = f"ec2-{dashed}.{_REGION}.compute.amazonaws.example"
  # The scanner fills these in only for a host it has actually scanned.
  scanned = bool(host.qids)
  scan_time = _LAST_FOUND if scanned else None
  tags = [_TAG_CONNECTOR]
  if host.worker:
    tags.append(_TAG_WORKERS)
  if host.internet_facing_tag:
    tags.append(_TAG_INTERNET_FACING)
  group_number = _SECURITY_GROUPS.index(host.security_group) + 1
  return {
      "id": asset_id,
      "name": host.name,
      "created": "2098-11-05T12:00:00Z",
      "modified": "2099-03-02T00:00:00Z",
      "type": "HOST",
      "tags": [{"id": tag_id, "name": name} for tag_id, name in tags],
      "sourceInfo": {
          "assetId": asset_id,
          "type": "EC_2",
          "firstDiscovered": "2098-11-05T12:00:00Z",
          "lastUpdated": "2099-03-02T00:00:00Z",
          "ec2InstanceTags": [
              {"key": key, "value": value}
              for key, value in (("Name", host.name), *host.ec2_tags)
          ],
          "reservationId": f"r-{n:017d}",
          "availabilityZone": f"{_REGION}a",
          "privateDnsName": private_dns,
          "publicDnsName": public_dns,
          "localHostname": private_dns,
          "instanceId": f"i-{n:017d}",
          "instanceType": "t3.medium",
          "createdDate": "2098-11-05T11:58:00Z",
          "instanceState": host.state,
          "groupId": f"sg-{group_number:017d}",
          "groupName": host.security_group,
          "spotInstance": "true" if host.worker else "false",
          "accountId": "000000000000",
          "subnetId": "subnet-00000000000000001",
          "vpcId": "vpc-00000000000000001",
          "region": _REGION,
          "zone": "VPC",
          "imageId": "ami-00000000000000001",
          "publicIpAddress": host.public_ip,
          "privateIpAddress": private_ip,
          "monitoringEnabled": "false",
      },
      "criticalityScore": str(host.criticality),
      "qwebHostId": str(800000000 + n) if scanned else None,
      "lastVulnScan": scan_time,
      "vulnsUpdated": scan_time,
      "informationGatheredUpdated": scan_time,
      "fqdn": public_dns or private_dns,
      "os": "Ubuntu Linux 20.04.6" if scanned else "Linux",
      "dnsHostName": public_dns or private_dns,
      "netbiosName": None,
      "networkGuid": f"00000000-0000-4000-8000-{n:012d}" if scanned else None,
      "address": host.public_ip or private_ip,
      "trackingMethod": "INSTANCE_ID",
      "cloudProvider": "AWS",
      "model": "t3.medium",
      "openPorts": [_open_port(port, proto) for port, proto in host.ports],
      "vulnerabilities": [
          {
              "qid": qid,
              "hostInstanceVulnId": str(first_detection_id + offset),
              "firstFound": _FIRST_FOUND,
              "lastFound": _LAST_FOUND,
          }
          for offset, qid in enumerate(host.qids)
      ],
      "networkInterfaces": [
          {
              "hostname": private_dns,
              "interfaceId": f"eni-{n:017d}",
              "interfaceName": None,
              "macAddress": f"02:00:00:00:00:{n:02x}",
              "address": private_ip,
          }
      ],
      "isDockerHost": "true" if host.docker else "false",
  }


def build_assets() -> list[dict[str, Any]]:
  """Returns the records of the asset export, one per host."""
  assets = []
  next_detection_id = _FIRST_DETECTION_ID
  for host in _HOSTS:
    assets.append(_asset(host, next_detection_id))
    next_detection_id += len(host.qids)
  return assets


def _cve_json(cve: _CveSpec) -> dict[str, Any]:
  """Returns the NVD record that the export embeds for a CVE.

  Args:
    cve: The CVE to describe.
  """
  record: dict[str, Any] = {
      "id": cve.cve_id,
      "sourceIdentifier": _CNA,
      # NVD writes milliseconds where the export's own field has microseconds.
      "published": cve.published[:-3],
      "lastModified": "2099-04-10T08:00:00.000",
      "vulnStatus": "Awaiting Analysis" if cve.cvss is None else "Analyzed",
      "cveTags": [],
      "descriptions": [{"lang": "en", "value": cve.description}],
      "metrics": {},
  }
  if cve.cvss is not None:
    vector = "N" if cve.attack_vector == "network" else "L"
    record["metrics"] = {
        "cvssMetricV31": [
            {
                "source": _NVD,
                "type": "Primary",
                "cvssData": {
                    "version": "3.1",
                    "vectorString": (
                        f"CVSS:3.1/AV:{vector}/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N"
                    ),
                    "attackVector": cve.attack_vector.upper(),
                    "baseScore": cve.cvss,
                },
            }
        ]
    }
  if cve.weaknesses is not None:
    record["weaknesses"] = [
        {
            "source": source,
            "type": "Primary" if source == _NVD else "Secondary",
            "description": [{"lang": "en", "value": value}],
        }
        for source, value in cve.weaknesses
    ]
  record["references"] = []
  for reference in cve.references:
    entry: dict[str, Any] = {"url": reference.url, "source": _CNA}
    if reference.tags is not None:
      entry["tags"] = list(reference.tags)
    record["references"].append(entry)
  if cve.known_exploit is not None:
    record["cisaExploitAdd"] = cve.known_exploit["dateAdded"]
    record["cisaActionDue"] = cve.known_exploit["dueDate"]
    record["cisaRequiredAction"] = cve.known_exploit["requiredAction"]
    record["cisaVulnerabilityName"] = cve.known_exploit["vulnerabilityName"]
  return record


def _enrichment(cve: _CveSpec) -> dict[str, Any]:
  """Returns the CVE-level keys of a vulns row, in the export's order.

  Args:
    cve: The CVE that the row pairs a detection with.
  """
  exploited = cve.known_exploit is not None
  digits = cve.cve_id.removeprefix("CVE-").replace("-", "")
  kev_published = None
  if cve.known_exploit is not None:
    kev_added = cve.known_exploit["dateAdded"]
    kev_published = f"{kev_added}T00:00:00"
  return {
      "cvss_base_score": cve.cvss,
      "id": f"00000000-0000-4000-8000-{digits:0>12}",
      "description": cve.description,
      "news_sources": "",
      "how_to_fix": cve.how_to_fix,
      "source_identifier": _CNA,
      "cve_json": _cve_json(cve),
      "news_json": [],
      "vuln_status": "Awaiting Analysis" if cve.cvss is None else "Analyzed",
      "publish_date": cve.published,
      "ease_of_exploit": "high" if exploited else "low",
      "x_tweet_json": None,
      "last_modified": "2099-04-10T08:00:00.000000",
      "likelihood_of_exploit": "high" if exploited else "low",
      "has_gh_exploit_poc": False,
      "attack_vector": cve.attack_vector,
      "known_exploit": exploited,
      "gh_exploit_poc_json": None,
      "cve_id": cve.cve_id,
      "epss": cve.epss,
      "known_exploit_publish_date": kev_published,
      "has_exploit_db_poc": False,
      "title": cve.title,
      "epss_percentile": cve.epss_percentile,
      "known_exploit_json": cve.known_exploit or {},
      "cogent_risk_score": cve.risk_score,
      "trending": False,
      "exploit_db_poc_json": None,
  }


def _vulns_row(
    detection: dict[str, str],
    asset: dict[str, Any],
    finding: _FindingSpec,
    cve: _CveSpec | None,
) -> dict[str, Any]:
  """Returns one row of the vulns export, keys in the export's order.

  Args:
    detection: The detection the row explains, from the asset record.
    asset: The host record the detection belongs to.
    finding: The detection's QID.
    cve: The CVE this row pairs the detection with, or None for a row that
      carries QID-level keys only.
  """
  return {
      **detection,
      **(_enrichment(cve) if cve else {}),
      # A copy, so that editing one row of the result edits one row only.
      "asset": copy.deepcopy(asset),
      "severity_level": finding.severity,
      "pci_flag": finding.pci_flag,
      "diagnosis": finding.diagnosis,
      "category": finding.category,
  }


def build_vulns(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Returns the rows of the vulns export that explain ``assets``.

  A detection of an explained QID yields one row per enriched CVE of that
  QID and one row without enrichment per CVE the export could not enrich.
  A QID with no CVE at all yields a single row without enrichment.

  Args:
    assets: The asset export, from ``build_assets``.
  """
  findings = {finding.qid: finding for finding in _FINDINGS}
  rows = []
  for asset in assets:
    for detection in asset["vulnerabilities"]:
      finding = findings.get(detection["qid"])
      if finding is None:
        continue
      bare_rows = finding.unenriched_cves if finding.cves else 1
      paired: list[_CveSpec | None] = [*finding.cves, *[None] * bare_rows]
      rows.extend(_vulns_row(detection, asset, finding, cve) for cve in paired)
  return rows


def write_fixtures(directory: pathlib.Path) -> None:
  """Writes ``assets.json`` and ``vulns.json`` into ``directory``.

  Args:
    directory: An existing directory.
  """
  assets = build_assets()
  exports = {"assets.json": assets, "vulns.json": build_vulns(assets)}
  for name, records in exports.items():
    text = json.dumps(records, indent=2) + "\n"
    (directory / name).write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
  write_fixtures(pathlib.Path(__file__).parent)
