/**
 * Blast Radius: the analyst-facing page.
 *
 * The page talks to three same-origin JSON endpoints (`GET /readyz`,
 * `POST /v1/analyze` and `GET /v1/hosts/{id}`) and renders what they return.
 * The shapes are the pydantic models in `blast_radius/models.py`; names such
 * as `AnalyzeResponse` or `HostGroup` in the comments below refer to them.
 *
 * Two rules shape the code.
 *
 * Untrusted text. Write-ups come from third parties and the brief comes from
 * a language model, so nothing in a response is ever parsed as markup. All
 * DOM is built by `el`, which turns strings into text nodes and refuses the
 * attributes that could carry script, style or a link target. The one place
 * that makes a link is `externalLink`, and only for URLs that parse as http
 * or https.
 *
 * Data in, DOM out. A rendering function takes a piece of a response and
 * returns nodes. None of them keeps state. The little state there is (the
 * response on screen, the request in flight, the control that opened the
 * drawer) lives in the closures at the bottom of the file.
 */

const READY_URL = "/readyz";
const ANALYZE_URL = "/v1/analyze";
const hostUrl = (id) => `/v1/hosts/${encodeURIComponent(id)}`;

/** `models.MAX_QUERY_CHARS`; the counter appears for the last thousand. */
const MAX_QUERY_CHARS = 8000;
const COUNTER_FROM_CHARS = 7000;

/** How many rows are rendered before a "show all" control takes over. */
const HOSTS_SHOWN = 25;
const CVES_SHOWN = 6;
const LIST_VALUES_SHOWN = 12;

const READINESS_RETRY_MS = 5000;

/** `EvidenceKind`, most actionable first, with the heading each goes under. */
const EVIDENCE_KINDS = [
  ["required_action", "Required action (CISA KEV)"],
  ["vendor_fix", "Vendor fix"],
  ["package_update", "Package update"],
  ["affected_versions", "Affected versions"],
  ["patch_reference", "Patch references"],
  ["advisory_reference", "Advisories"],
];

const DOC_TYPE_LABELS = { cve: "CVE description", qid: "QID write-up" };

const FAILURE_TITLES = {
  network: "The service could not be reached",
  invalid: "The query was rejected",
  not_ready: "The service is not ready",
  not_found: "The service has no such record",
  server: "The request failed on the server",
  unexpected: "The service returned something this page cannot read",
};

const FAILURE_HINTS = {
  network: "Check that the service is running, then try again.",
  not_ready: "The status at the top of the page changes when it is.",
  server: "Quote the request id when reporting this; it finds the log line.",
};

const KEV_TITLE = "Known exploited (CISA KEV)";

const NUMBER = new Intl.NumberFormat("en-GB");
const DATE = new Intl.DateTimeFormat("en-GB", { dateStyle: "medium" });
const DATE_TIME = new Intl.DateTimeFormat("en-GB", {
  dateStyle: "medium",
  timeStyle: "short",
});

// ---------------------------------------------------------------------------
// DOM and formatting helpers
// ---------------------------------------------------------------------------

/**
 * Attributes `el` refuses. Inline handlers and inline styles would be blocked
 * by the Content-Security-Policy anyway; link and resource targets are kept
 * out so that `externalLink` stays the only way to point anywhere.
 */
const REFUSED_ATTRIBUTE =
  /^(on.*|style|srcdoc|href|src|action|formaction)$/i;

/**
 * Creates an element.
 *
 * @param {string} tag The element name.
 * @param {object} props `class` and `text` set the class and the text
 *   content, `on` maps event names to listeners, and every other key is set
 *   as an attribute. `null`, `undefined` and `false` values are skipped and
 *   `true` sets an empty attribute.
 * @param {...(Node|string|number|null|false|Array)} children Appended in
 *   order, arrays flattened. Strings become text nodes; empty values are
 *   skipped, so `condition && node` reads naturally at a call site.
 * @returns {HTMLElement}
 */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) {
      continue;
    }
    if (name === "class") {
      node.className = value;
    } else if (name === "text") {
      node.textContent = value;
    } else if (name === "on") {
      for (const [type, listener] of Object.entries(value)) {
        node.addEventListener(type, listener);
      }
    } else if (REFUSED_ATTRIBUTE.test(name)) {
      throw new Error(`el() does not set the "${name}" attribute`);
    } else {
      node.setAttribute(name, value === true ? "" : String(value));
    }
  }
  const isPresent = (child) =>
    child !== null && child !== undefined && child !== false && child !== "";
  node.append(...children.flat(Infinity).filter(isPresent));
  return node;
}

/** Returns a copy of the static content of one of the page's templates. */
function fromTemplate(template) {
  return [...template.content.cloneNode(true).children];
}

/** Returns e.g. "1 host" or "1,273 hosts". */
function plural(count, singular, many = `${singular}s`) {
  return `${NUMBER.format(count)} ${count === 1 ? singular : many}`;
}

/** Returns a score with a fixed number of decimals. */
function formatScore(value, digits = 2) {
  return value.toFixed(digits);
}

/** Returns a duration in the unit a reader would use for it. */
function formatDuration(ms) {
  if (ms < 1) {
    return "<1 ms";
  }
  if (ms < 1000) {
    return `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`;
  }
  return `${(ms / 1000).toFixed(2)} s`;
}

/**
 * Returns a `<time>` for an ISO timestamp in the reader's locale, with the
 * exact value on hover. Text that is not a date is shown as it came.
 *
 * @param {string} iso The timestamp.
 * @param {Intl.DateTimeFormat} [format] `DATE` where the day is enough.
 */
function timestamp(iso, format = DATE_TIME) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return el("span", { text: iso });
  }
  return el("time", { datetime: iso, title: iso, text: format.format(date) });
}

/** Returns the share `value` is of `max`, clamped to 0..1. */
function shareOf(value, max) {
  return max > 0 ? Math.min(Math.max(value / max, 0), 1) : 0;
}

/** Returns "QID 38919" or the CVE id of a source, evidence item or chunk. */
function docLabel(doc) {
  return doc.doc_type === "qid" ? `QID ${doc.doc_id}` : doc.doc_id;
}

/** Returns a snake_case name, such as an `EvidenceKind`, as words. */
function words(name) {
  return name.replaceAll("_", " ");
}

// ---------------------------------------------------------------------------
// Links
// ---------------------------------------------------------------------------

/** Returns `text` as a URL if it is an absolute http(s) URL, else null. */
function httpUrl(text) {
  let url;
  try {
    url = new URL(text);
  } catch {
    return null;
  }
  return url.protocol === "http:" || url.protocol === "https:" ? url : null;
}

/** Returns a link that opens `url` in a new tab and tells it nothing. */
function externalLink(url, label) {
  const link = el("a", {
    rel: "noopener noreferrer nofollow",
    target: "_blank",
    text: label,
  });
  link.href = url.href;
  return link;
}

/**
 * Returns `text` as nodes, with `reference` made a link where it occurs.
 *
 * A reference that is not an http(s) URL stays the plain text it already is
 * inside `text`, or is appended as plain text when `text` does not hold it.
 */
function textWithReference(text, reference) {
  if (!reference) {
    return [text];
  }
  const url = httpUrl(reference);
  const at = text.indexOf(reference);
  if (at < 0) {
    const shown = url
      ? externalLink(url, reference)
      : el("span", { class: "mono", text: reference });
    return [text, " ", shown];
  }
  if (!url) {
    return [text];
  }
  return [
    text.slice(0, at),
    externalLink(url, reference),
    text.slice(at + reference.length),
  ];
}

// ---------------------------------------------------------------------------
// Quotes
// ---------------------------------------------------------------------------

/**
 * Collapses every run of whitespace in `text` to one space and trims it.
 *
 * @returns {{normalised: string, offsets: number[]}} `offsets[i]` is the
 *   index in `text` of the character that became `normalised[i]`.
 */
function normaliseWithOffsets(text) {
  let normalised = "";
  const offsets = [];
  let gapAt = -1;
  for (let i = 0; i < text.length; i += 1) {
    if (/\s/.test(text[i])) {
      gapAt = gapAt < 0 ? i : gapAt;
      continue;
    }
    if (gapAt >= 0 && normalised) {
      normalised += " ";
      offsets.push(gapAt);
    }
    gapAt = -1;
    normalised += text[i];
    offsets.push(i);
  }
  return { normalised, offsets };
}

/**
 * Finds `quote` in `text` the way the verifier accepts it: exactly, or else
 * with whitespace runs treated as single spaces.
 *
 * @returns {{start: number, end: number}|null} Offsets into `text`.
 */
function locateQuote(text, quote) {
  const exact = quote ? text.indexOf(quote) : -1;
  if (exact >= 0) {
    return { start: exact, end: exact + quote.length };
  }
  const needle = normaliseWithOffsets(quote).normalised;
  if (!needle) {
    return null;
  }
  const { normalised, offsets } = normaliseWithOffsets(text);
  const at = normalised.indexOf(needle);
  if (at < 0) {
    return null;
  }
  return { start: offsets[at], end: offsets[at + needle.length - 1] + 1 };
}

/** Returns `text` as nodes, with the `span` from `locateQuote` marked. */
function highlighted(text, span) {
  if (!span) {
    return [text];
  }
  return [
    text.slice(0, span.start),
    el("mark", { text: text.slice(span.start, span.end) }),
    text.slice(span.end),
  ];
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/** A request that did not produce a usable answer. */
class ApiFailure extends Error {
  /**
   * @param {string} kind A key of `FAILURE_TITLES`.
   * @param {string|null} detail What the service said, if anything.
   * @param {string|null} requestId The `X-Request-ID` to find the log line.
   */
  constructor(kind, detail = null, requestId = null) {
    super(FAILURE_TITLES[kind]);
    this.kind = kind;
    this.detail = detail;
    this.requestId = requestId;
  }
}

/** Returns any thrown value as an `ApiFailure` the page can show. */
function asFailure(error) {
  if (error instanceof ApiFailure) {
    return error;
  }
  return new ApiFailure("unexpected", String(error?.message ?? error));
}

/**
 * Sends a request and decodes a JSON body, whatever the status.
 *
 * @returns {Promise<{status: number, ok: boolean, body: *, requestId:
 *   string|null}>} `body` is null when the reply was not JSON.
 * @throws {ApiFailure} If the service cannot be reached.
 * @throws {DOMException} An `AbortError`, if `init.signal` fired.
 */
async function requestJson(url, init = {}) {
  let response;
  let body = null;
  try {
    response = await fetch(url, {
      ...init,
      headers: { Accept: "application/json", ...init.headers },
    });
    body = await response.json();
  } catch (error) {
    if (error.name === "AbortError") {
      throw error;
    }
    if (!response) {
      throw new ApiFailure("network");
    }
  }
  return {
    status: response.status,
    ok: response.ok,
    body,
    requestId: response.headers.get("X-Request-ID"),
  };
}

/** Returns the human-readable part of an error body, or null. */
function detailOf(body) {
  if (body === null || typeof body !== "object") {
    return null;
  }
  if (typeof body.detail === "string") {
    return body.detail;
  }
  if (Array.isArray(body.detail)) {
    // FastAPI's validation errors: one object with a `msg` per problem.
    const messages = body.detail
      .map((problem) => problem?.msg)
      .filter((message) => typeof message === "string");
    return messages.length > 0 ? messages.join(" ") : null;
  }
  return typeof body.reason === "string" ? body.reason : null;
}

/** Returns the `ApiFailure` that describes an unsuccessful reply. */
function failureFrom(reply) {
  const kinds = { 404: "not_found", 422: "invalid", 503: "not_ready" };
  const fallback = reply.status >= 500 ? "server" : "unexpected";
  return new ApiFailure(
    kinds[reply.status] ?? fallback,
    detailOf(reply.body) ?? `The service answered HTTP ${reply.status}.`,
    reply.requestId ?? reply.body?.request_id ?? null,
  );
}

/** Returns whether `body` has the outline of an `AnalyzeResponse`. */
function isAnalyzeResponse(body) {
  const lists = [
    "matches",
    "groups",
    "hosts",
    "inactive_hosts",
    "fix_evidence",
    "context",
    "caveats",
    "notices",
    "trace",
  ];
  return (
    body !== null &&
    typeof body === "object" &&
    (body.status === "matched" || body.status === "no_match") &&
    typeof body.parsed === "object" &&
    typeof body.meta === "object" &&
    lists.every((name) => Array.isArray(body[name]))
  );
}

/** Runs the pipeline for `query` and returns the `AnalyzeResponse`. */
async function analyse(query, signal) {
  const reply = await requestJson(ANALYZE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });
  if (!reply.ok) {
    throw failureFrom(reply);
  }
  if (!isAnalyzeResponse(reply.body)) {
    throw new ApiFailure("unexpected", null, reply.requestId);
  }
  return reply.body;
}

/** Returns the `HostDetail` of one host. */
async function fetchHost(id, signal) {
  const reply = await requestJson(hostUrl(id), { signal });
  if (!reply.ok) {
    throw failureFrom(reply);
  }
  if (typeof reply.body?.host !== "object") {
    throw new ApiFailure("unexpected", null, reply.requestId);
  }
  return reply.body;
}

/**
 * Returns the service's `Readiness`. Never throws: a service that cannot be
 * asked is reported as `{status: "unreachable", reason}`, a state only this
 * page has.
 */
async function fetchReadiness() {
  try {
    const reply = await requestJson(READY_URL);
    const status = reply.body?.status;
    if (status === "ready" || status === "not_ready") {
      return reply.body;
    }
    return {
      status: "unreachable",
      reason: `The readiness check answered HTTP ${reply.status}.`,
    };
  } catch {
    return { status: "unreachable", reason: "The service did not answer." };
  }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/** Returns a word in a box; `tone` picks the colour that repeats the word. */
function badge(text, tone = null, title = null) {
  const className = tone ? `badge badge-${tone}` : "badge";
  return el("span", { class: className, title, text });
}

/** Returns the Qualys severity as five pips and as words. */
function severity(level) {
  if (!Number.isInteger(level)) {
    return el("span", { class: "muted", text: "severity not given" });
  }
  const pips = [1, 2, 3, 4, 5].map((n) =>
    el("span", { class: n <= level ? "pip is-on" : "pip" }),
  );
  return el(
    "span",
    { class: "severity" },
    el(
      "span",
      { class: "pips", "data-level": level, "aria-hidden": "true" },
      pips,
    ),
    `severity ${level} of 5`,
  );
}

/** Returns a priority as a number beside a bar that is its share of `max`. */
function priorityMeter(value, max) {
  const fill = el("span", { class: "meter-fill" });
  fill.style.setProperty("--fill", shareOf(value, max).toFixed(4));
  return el(
    "span",
    { class: "meter" },
    el("span", { class: "meter-track", "aria-hidden": "true" }, fill),
    el("span", { class: "unit", text: "priority " }),
    el("span", { class: "meter-value", text: formatScore(value) }),
  );
}

/** Returns an attack vector as a badge; `network` is the one to stand out. */
function vectorBadge(vector, count = null) {
  const label = `AV ${words(vector ?? "unknown")}`;
  return badge(
    count === null ? label : `${label} × ${NUMBER.format(count)}`,
    vector === "network" ? "strong" : "quiet",
    "Attack vector",
  );
}

/**
 * Returns a citable id. It opens the source in the drawer when the response
 * carries that source, and is an inert label when it does not.
 *
 * @param {object} citable
 * @param {string} citable.id A `ContextItem.id`, shown as it is.
 * @param {object|undefined} citable.source The `ContextItem` to open.
 * @param {string|null} [citable.quote] A quote to highlight in the source.
 * @param {string|null} [citable.note] Shown after the id, e.g. a score.
 * @param {boolean} [citable.flagged] Whether verification failed here.
 * @param {object} actions See `main`.
 */
function citeChip(citable, actions) {
  const { id, source, quote = null, note = null, flagged = false } = citable;
  const noteNode = note && el("span", { class: "cite-note", text: note });
  if (!source) {
    const title = "This id is not among the sources";
    return el("span", { class: "cite-chip is-static", title }, id, noteNode);
  }
  const open = (event) =>
    actions.openSource(source, quote, event.currentTarget);
  return el(
    "button",
    {
      type: "button",
      class: flagged ? "cite-chip is-flagged" : "cite-chip",
      "aria-haspopup": "dialog",
      "aria-expanded": "false",
      on: { click: open },
    },
    el("span", { class: "visually-hidden", text: "Open source " }),
    id,
    noteNode,
  );
}

/** Returns a titled callout: `notice`, `limits`, `waiting` or `error`. */
function callout(tone, title, ...children) {
  return el(
    "div",
    { class: `callout callout-${tone}` },
    el("h3", { class: "callout-title", text: title }),
    children,
  );
}

/** Returns the caveats as what they are: the edges of what the data knows. */
function limitsCallout(caveats, title = "Limits of the data") {
  const items = caveats.map((caveat) => el("li", { text: caveat }));
  return callout("limits", title, el("ul", { class: "limit-list" }, items));
}

/** Returns the service's operational notes, e.g. why there is no brief. */
function noticesCallout(notices) {
  const paragraphs = notices.map((notice) => el("p", { text: notice }));
  return callout("notice", "Notices", paragraphs);
}

/**
 * Returns a button that shows and hides a panel. The panel's content is
 * built the first time it opens, so a closed disclosure costs nothing.
 *
 * @param {object} options
 * @param {string} options.id Id for the panel; must be unique on the page.
 * @param {string} options.label The button's text.
 * @param {string|null} [options.note] Quieter text after the label.
 * @param {function(): (Node|Node[])} options.render Builds the content.
 */
function disclosure({ id, label, note = null, render }) {
  const panel = el("div", { class: "disclosure-panel", id, hidden: true });
  const noteNode =
    note && el("span", { class: "disclosure-note", text: ` ${note}` });
  const button = el(
    "button",
    {
      type: "button",
      class: "disclosure",
      "aria-expanded": "false",
      "aria-controls": id,
      on: { click: () => togglePanel(button, panel, render) },
    },
    el("span", { class: "chevron", "aria-hidden": "true" }),
    el("span", {}, label, noteNode),
  );
  return el("div", {}, button, panel);
}

/** Flips a disclosure, building the panel's content on first use. */
function togglePanel(button, panel, render) {
  const open = button.getAttribute("aria-expanded") !== "true";
  if (open && !panel.hasChildNodes()) {
    panel.append(...[render()].flat());
  }
  button.setAttribute("aria-expanded", String(open));
  panel.hidden = !open;
}

/**
 * Fills `container` with the first `shown` items and returns the control
 * that renders the rest on demand, or null when everything already fits.
 * `container` needs an id for the control to point at.
 */
function fillCapped(container, items, renderItem, shown, noun) {
  container.append(...items.slice(0, shown).map(renderItem));
  if (items.length <= shown) {
    return null;
  }
  const allLabel = `Show all ${plural(items.length, noun)}`;
  const toggle = () => {
    const open = button.getAttribute("aria-expanded") !== "true";
    if (open) {
      container.append(...items.slice(shown).map(renderItem));
    } else {
      [...container.children].slice(shown).forEach((node) => node.remove());
    }
    button.setAttribute("aria-expanded", String(open));
    button.textContent = open ? `Show the first ${shown} only` : allLabel;
  };
  const button = el("button", {
    type: "button",
    class: "more-button",
    "aria-expanded": "false",
    "aria-controls": container.id,
    text: allLabel,
    on: { click: toggle },
  });
  return button;
}

/** Returns one value of a trace summary or of the provenance as text. */
function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "none";
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  if (Array.isArray(value)) {
    const shown = value.slice(0, LIST_VALUES_SHOWN).map(formatValue);
    const hidden = value.length - shown.length;
    const list = shown.join(", ") || "none";
    return hidden > 0 ? `${list}, and ${NUMBER.format(hidden)} more` : list;
  }
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

/**
 * Returns a definition list.
 *
 * @param {Array<[string, (Node|string)]>} pairs Name and value, in order.
 * @param {string} [layout] `kv` for a two-column grid, `kv-inline` to flow.
 */
function keyValues(pairs, layout = "kv") {
  const rows = pairs.map(([name, value]) => [
    el("dt", { text: name }),
    el("dd", {}, value),
  ]);
  // The grid lays out dt and dd directly; the inline form wraps pair by pair.
  return el(
    "dl",
    { class: layout },
    layout === "kv" ? rows : rows.map((row) => el("div", {}, row)),
  );
}

/**
 * Returns a row of large figures, each over the words that say what it
 * counts.
 *
 * @param {Array<{value: string, label: string, strong?: boolean}>} stats
 */
function statRow(stats) {
  return el(
    "dl",
    { class: "headline" },
    stats.map(({ value, label, strong = false }) =>
      el(
        "div",
        { class: strong ? "stat is-strong" : "stat" },
        el("dt", { class: "stat-label", text: label }),
        el("dd", { class: "stat-value", text: value }),
      ),
    ),
  );
}

/** Returns a table head from `[name, class]` pairs; `num` aligns right. */
function tableHead(columns) {
  const cells = columns.map(([name, modifier]) =>
    el("th", { scope: "col", class: modifier, text: name }),
  );
  return el("thead", {}, el("tr", {}, cells));
}

/**
 * Returns the column names above a list whose rows are laid out on the same
 * grid. They are decoration: every row carries its own labels for a screen
 * reader, and on a narrow screen the names are dropped.
 */
function columnNames(className, columns) {
  const cells = columns.map(([name, modifier]) =>
    el("span", { class: modifier, text: name }),
  );
  return el("div", { class: className, "aria-hidden": "true" }, cells);
}

/** Returns a SHA-256 shortened for reading, whole on hover. */
function hash(value) {
  if (!value) {
    return "none";
  }
  const short = `${value.slice(0, 12)}…`;
  return el("span", { class: "hash", title: value, text: short });
}

/** Returns a small-caps heading followed by content, as one block. */
function block(className, title, ...content) {
  return el(
    "div",
    { class: className },
    el("h3", { class: "block-title", text: title }),
    content,
  );
}

/**
 * Returns one ruled section of the report.
 *
 * @param {object} options
 * @param {string} options.id Prefix for the heading's id.
 * @param {string} options.title The name in the rail.
 * @param {string|null} [options.index] Two digits shown above the name.
 * @param {Array<Node|string>} [options.meta] Small print under the name.
 * @param {Array<Node|null|false>} options.body The section's content.
 */
function section({ id, title, index = null, meta = [], body }) {
  const headingId = `${id}-heading`;
  const indexNode =
    index &&
    el("p", { class: "rail-index", "aria-hidden": "true", text: index });
  return el(
    "section",
    { class: "report-section", "aria-labelledby": headingId },
    el(
      "div",
      { class: "rail" },
      indexNode,
      el("h2", { class: "rail-title", id: headingId, text: title }),
      meta.map((line) => el("p", { class: "rail-meta" }, line)),
    ),
    el("div", { class: "section-body" }, body),
  );
}

// ---------------------------------------------------------------------------
// Report: the answer
// ---------------------------------------------------------------------------

/** Returns "Matched" or "No match" for the rail. */
function verdict(status) {
  const text = status === "matched" ? "Matched" : "No match";
  return el("span", { class: "verdict", "data-status": status, text });
}

/** Returns the query as the service received it, clamped to two lines. */
function renderQueryEcho(query) {
  return el(
    "p",
    { class: "query-echo" },
    el("span", { class: "block-title", text: "Query" }),
    el("span", { class: "query-echo-text", title: query, text: query }),
  );
}

/** Returns "5 of 6 claims verified", with one pip per claim, in order. */
function renderTally(verification, claims) {
  if (!verification) {
    return el("p", { class: "tally" }, badge("not verified", "warn"));
  }
  const pips = claims.map((claim) =>
    el("span", { class: claim.verified ? "tally-pip" : "tally-pip is-off" }),
  );
  const verified = NUMBER.format(verification.verified_claims);
  return el(
    "p",
    { class: "tally" },
    el("span", { class: "tally-pips", "aria-hidden": "true" }, pips),
    `${verified} of ${plural(verification.total_claims, "claim")} verified`,
  );
}

/** Returns one citation: its chip, then the quote it rests on. */
function renderCitation(citation, sources, actions) {
  const failed = citation.verified === false;
  const chip = citeChip(
    {
      id: citation.source_id,
      source: sources.get(citation.source_id),
      quote: citation.quote,
      note: failed ? "quote not found" : null,
      flagged: failed,
    },
    actions,
  );
  return el(
    "li",
    { class: "citation" },
    chip,
    el("span", { class: "quote", text: citation.quote }),
  );
}

/** Returns the name of a `verified` value: true, false, or not set. */
function verificationState(verified) {
  if (verified === null) {
    return "unchecked";
  }
  return verified ? "verified" : "unverified";
}

/**
 * Returns one claim of the brief. A claim that failed verification is kept
 * whole and readable, flagged, with the verifier's reasons under it.
 */
function renderClaim(claim, sources, actions) {
  const state = verificationState(claim.verified);
  const flags = {
    verified: el("span", { class: "visually-hidden", text: "Verified. " }),
    unverified: el("p", { class: "claim-flag", text: "Unverified" }),
    unchecked: el("p", { class: "claim-flag", text: "Not checked" }),
  };
  const citations = claim.citations.map((citation) =>
    renderCitation(citation, sources, actions),
  );
  const problems = claim.problems.map((problem) => el("li", { text: problem }));
  return el(
    "li",
    { class: state === "unverified" ? "claim is-unverified" : "claim" },
    el("span", {
      class: "claim-marker",
      "data-state": state,
      "aria-hidden": "true",
    }),
    el(
      "div",
      {},
      flags[state],
      el("p", { class: "claim-text", text: claim.text }),
      citations.length > 0 && el("ul", { class: "citations" }, citations),
      problems.length > 0 &&
        el(
          "div",
          { class: "problems" },
          el("p", { class: "problems-title", text: "Why it is flagged" }),
          el("ul", {}, problems),
        ),
    ),
  );
}

/** Returns the language model's brief: summary, claims, its own caveats. */
function renderBrief(response, sources, actions) {
  const { answer, verification, meta } = response;
  const writer = meta.llm_model ?? "a language model";
  const unknown = verification?.unknown_identifiers ?? [];
  const claims = answer.claims.map((claim) =>
    renderClaim(claim, sources, actions),
  );
  const caveats = answer.caveats.map((caveat) => el("li", { text: caveat }));
  return el(
    "div",
    { class: "brief" },
    el(
      "div",
      { class: "brief-head" },
      el(
        "div",
        {},
        el("h3", { class: "block-title", text: "Written brief" }),
        el("p", {
          class: "brief-byline",
          text:
            `Written by ${writer}. Every quote is checked against its` +
            " source.",
        }),
      ),
      renderTally(verification, answer.claims),
    ),
    el("p", { class: "brief-summary", text: answer.summary }),
    claims.length > 0 && el("ol", { class: "claims" }, claims),
    unknown.length > 0 &&
      el(
        "p",
        { class: "unknown-ids" },
        "Identifiers in the brief that occur in no source: ",
        el("span", { class: "mono", text: unknown.join(", ") }),
      ),
    caveats.length > 0 &&
      el(
        "div",
        {},
        el("h4", { class: "block-title", text: "The brief's own caveats" }),
        el("ul", { class: "brief-caveats" }, caveats),
      ),
  );
}

/** Returns what stands in for the brief when the response has none. */
function renderNoBrief(notices) {
  const reasons =
    notices.length > 0
      ? notices
      : ["The service returned no written brief for this response."];
  return block(
    "brief",
    "Written brief",
    reasons.map((reason) => el("p", { class: "brief-empty", text: reason })),
  );
}

/** Returns the first section of a matched response. */
function renderAnswer(response, sources, actions) {
  const { answer, notices, caveats, meta } = response;
  const writer = meta.llm_model ?? "a language model";
  return section({
    id: "answer",
    index: "01",
    title: "Answer",
    meta: [
      verdict(response.status),
      answer ? `Brief by ${writer}` : "No written brief",
    ],
    body: [
      renderQueryEcho(response.query),
      el("p", { class: "computed-summary", text: response.summary }),
      // Without a brief, the notices say why, in the brief's place.
      answer
        ? renderBrief(response, sources, actions)
        : renderNoBrief(notices),
      answer && notices.length > 0 && noticesCallout(notices),
      caveats.length > 0 && limitsCallout(caveats),
    ],
  });
}

/** Returns what step 1 looked up and searched for, to help a rephrase. */
function renderSearched(parsed) {
  const identifiers = [
    ...parsed.cve_ids,
    ...parsed.qids.map((qid) => `QID ${qid}`),
  ];
  const pairs = [];
  if (identifiers.length > 0) {
    pairs.push(["Looked up", identifiers.join(", ")]);
  }
  if (parsed.search_queries.length > 0) {
    pairs.push(["Searched for", parsed.search_queries.join(" · ")]);
  }
  return pairs.length > 0
    ? block("searched", "What was tried", keyValues(pairs))
    : null;
}

/**
 * Returns the first section of a `no_match` response. The point it has to
 * land is that no match is not a clean bill of health, so the page says so
 * in its own words, and the service's caveats, with the numbers, follow.
 */
function renderNoMatch(response) {
  const { notices, caveats } = response;
  const lede =
    `${response.summary} A query can only match what the scanner's` +
    " write-ups describe, and they describe a part of what it detects.";
  return section({
    id: "answer",
    index: "01",
    title: "Answer",
    meta: [verdict(response.status)],
    body: [
      renderQueryEcho(response.query),
      el("p", {
        class: "no-match-title",
        text: "No match, which is not the same as not affected.",
      }),
      el("p", { class: "no-match-lede", text: lede.trim() }),
      caveats.length > 0 &&
        limitsCallout(caveats, "What the data cannot rule out"),
      notices.length > 0 && noticesCallout(notices),
      renderSearched(response.parsed),
    ],
  });
}

// ---------------------------------------------------------------------------
// Report: what matched
// ---------------------------------------------------------------------------

/** Returns one `CveSummary` as a row: id, title, then flags and scores. */
function renderCve(cve, isHit) {
  const score = (text) => el("span", { class: "cve-score", text });
  return el(
    "li",
    { class: "cve" },
    el("span", { class: "cve-id", text: cve.cve_id }),
    el("span", { class: "cve-title", title: cve.title, text: cve.title }),
    el(
      "div",
      { class: "cve-facts" },
      isHit && badge("hit by the query", "accent"),
      cve.known_exploited && badge("KEV", "danger", KEV_TITLE),
      vectorBadge(cve.attack_vector),
      cve.cvss !== null && score(`CVSS ${formatScore(cve.cvss, 1)}`),
      cve.epss_percentile !== null &&
        score(`EPSS percentile ${formatScore(cve.epss_percentile)}`),
    ),
  );
}

/** Returns the KEV and attack-vector badges that sum up a bundle of CVEs. */
function renderCveTotals(cves) {
  const exploited = cves.filter((cve) => cve.known_exploited).length;
  const vectors = new Map();
  for (const cve of cves) {
    vectors.set(cve.attack_vector, (vectors.get(cve.attack_vector) ?? 0) + 1);
  }
  return el(
    "div",
    { class: "badge-row" },
    exploited > 0 &&
      badge(`KEV × ${NUMBER.format(exploited)}`, "danger", KEV_TITLE),
    [...vectors].map(([vector, count]) => vectorBadge(vector, count)),
  );
}

/**
 * Returns a match's CVEs, the ones the query hit first. A QID can bundle
 * hundreds, so the list is capped.
 */
function renderCves(match, index) {
  const hit = new Set(match.matched_cve_ids);
  const cves = [
    ...match.cves.filter((cve) => hit.has(cve.cve_id)),
    ...match.cves.filter((cve) => !hit.has(cve.cve_id)),
  ];
  const count =
    cves.length === 0
      ? "No CVE attached"
      : `${plural(cves.length, "CVE")}, ${NUMBER.format(hit.size)} hit by` +
        " the query";
  const list = el("ul", { class: "cve-list", id: `match-${index}-cves` });
  const renderItem = (cve) => renderCve(cve, hit.has(cve.cve_id));
  const more = fillCapped(list, cves, renderItem, CVES_SHOWN, "CVE");
  return [
    el(
      "div",
      { class: "match-line" },
      el("span", { class: "match-line-label", text: count }),
      // One CVE's badges are on its own row; a bundle needs the totals.
      cves.length > 1 && renderCveTotals(cves),
    ),
    cves.length > 0 && list,
    more,
  ];
}

/** Returns the chips of the chunks that produced a search match. */
function renderMatchedText(match, sources, actions) {
  const chips = match.chunks.map((scored) => {
    const id = `c${scored.chunk.id}`;
    const score = scored.rerank_score ?? scored.dense_score;
    // A chunk is a complete source in itself, should the context cap have
    // dropped it from `context`.
    const source = sources.get(id) ?? { ...scored.chunk, id };
    const note = score === null ? null : formatScore(score);
    return citeChip({ id, source, note }, actions);
  });
  return el(
    "div",
    { class: "match-line" },
    el("span", { class: "match-line-label", text: "Matched text" }),
    chips,
  );
}

/** Returns one `QidMatch` as a card. */
function renderMatch(match, index, sources, actions) {
  const score =
    match.score === null ? "" : ` · score ${formatScore(match.score)}`;
  const how =
    match.matched_by === "identifier"
      ? "matched by identifier"
      : `matched by search${score}`;
  return el(
    "li",
    { class: "match" },
    el(
      "div",
      { class: "match-head" },
      el("span", { class: "match-qid", text: `QID ${match.qid}` }),
      el("span", { class: "match-how", text: how }),
    ),
    match.label && el("h3", { class: "match-label", text: match.label }),
    el(
      "div",
      { class: "match-facts" },
      match.category && el("span", { text: match.category }),
      severity(match.severity),
    ),
    renderCves(match, index),
    match.chunks.length > 0 && renderMatchedText(match, sources, actions),
  );
}

/** Returns the section with one card per matched QID. */
function renderMatches(response, sources, actions) {
  const { matches } = response;
  const cards = matches.map((match, index) =>
    renderMatch(match, index, sources, actions),
  );
  return section({
    id: "matches",
    index: "02",
    title: "What matched",
    meta: [
      `${plural(matches.length, "scanner check")}. Hosts attach to checks,` +
        " so each CVE is reported under its QID.",
    ],
    body: [el("ul", { class: "match-grid" }, cards)],
  });
}

// ---------------------------------------------------------------------------
// Report: the blast radius
// ---------------------------------------------------------------------------

/** Returns the four counts that size the problem. */
function renderHeadline(response) {
  const facing = response.hosts.filter(
    (ranked) => ranked.host.internet_facing,
  ).length;
  const stat = (count, label, strong = false) => ({
    value: NUMBER.format(count),
    label,
    strong,
  });
  return statRow([
    stat(response.hosts.length, "running affected hosts"),
    stat(facing, "of them internet-facing", facing > 0),
    stat(response.groups.length, "groups to work through"),
    stat(response.inactive_hosts.length, "inactive, listed but not ranked"),
  ]);
}

/** Returns whether a host faces the internet, as a badge or a quiet word. */
function exposure(host) {
  return host.internet_facing
    ? badge("internet-facing", "exposed")
    : el("span", { class: "muted", text: "internal" });
}

/** Returns a host's addresses, public first. */
function addresses(host) {
  const lines = [
    host.public_ip && ["public", host.public_ip],
    host.private_ip && ["private", host.private_ip],
  ].filter(Boolean);
  if (lines.length === 0) {
    return el("span", { class: "muted", text: "no address" });
  }
  return el(
    "div",
    { class: "addresses" },
    lines.map(([kind, ip]) =>
      el("div", {}, ip, " ", el("span", { class: "address-kind", text: kind })),
    ),
  );
}

/** Returns the host's name as the control that opens its detail. */
function hostLink(host, actions) {
  return el("button", {
    type: "button",
    class: "host-link",
    "aria-haspopup": "dialog",
    "aria-expanded": "false",
    text: host.name,
    on: { click: (event) => actions.openHost(host, event.currentTarget) },
  });
}

/**
 * Returns "why this order": the arithmetic behind one host's priority, from
 * its `PriorityFactors`. The export's own risk score comes last and is
 * labelled as the comparison it is, because it is not an input.
 */
function renderWhy(ranked) {
  const { factors } = ranked;
  const value = (text) => el("span", { class: "factor-value", text });
  const part = (...content) => el("span", { class: "why-part" }, content);
  const others = ranked.qids.filter((qid) => qid !== factors.driving_qid);
  const risk = factors.cogent_risk_score;
  return el(
    "p",
    { class: "host-detail" },
    el("span", { class: "visually-hidden", text: "Why this order: " }),
    el(
      "span",
      { class: "why-part is-result" },
      "priority ",
      value(formatScore(ranked.priority)),
      " = threat ",
      value(formatScore(factors.threat)),
      " × exposure ",
      value(formatScore(factors.exposure)),
    ),
    part(
      "threat from severity ",
      value(formatScore(factors.severity)),
      ", EPSS percentile ",
      value(formatScore(factors.epss_percentile)),
      ", KEV ",
      value(factors.known_exploited ? "yes" : "no"),
    ),
    part(
      "exposure from ",
      value(factors.internet_facing ? "internet-facing" : "internal"),
      ", criticality ",
      value(factors.criticality),
    ),
    part(
      "driven by QID ",
      value(factors.driving_qid),
      others.length > 0 && [", also matched QID ", value(others.join(", "))],
    ),
    part(
      "export's own risk score, for comparison ",
      value(risk === null ? "none" : formatScore(risk, 1)),
    ),
  );
}

/**
 * Returns one host as a record of two lines: its facts in columns, then a
 * line of detail. Records stack into a column on a narrow screen, where a
 * table of the same facts would have to scroll sideways.
 *
 * @param {object} host A `Host`.
 * @param {Node} priority What goes in the priority column.
 * @param {Node} detail The second line.
 * @param {object} actions See `main`.
 */
function renderHostRecord(host, priority, detail, actions) {
  const criticality = [
    el("span", { class: "unit", text: "criticality " }),
    host.criticality,
  ];
  return el(
    "li",
    { class: "host" },
    el(
      "div",
      { class: "host-fields" },
      el("div", { class: "host-name" }, hostLink(host, actions)),
      el("div", { class: "host-priority" }, priority),
      el("div", { text: host.state.toLowerCase() }),
      el("div", {}, criticality),
      el("div", {}, exposure(host)),
      el("div", { class: "host-addresses" }, addresses(host)),
    ),
    detail,
  );
}

/** Returns the column names that sit above a list of host records. */
function hostColumnNames() {
  return columnNames("hosts-header", [
    ["Host"],
    ["Priority"],
    ["State"],
    ["Criticality"],
    ["Exposure"],
    ["Addresses"],
  ]);
}

/** Returns the hosts of one group, capped, with a control for the rest. */
function renderGroupHosts(group, index, ranking, actions) {
  const hosts = group.host_ids
    .map((id) => ranking.hostsById.get(id))
    .filter(Boolean);
  const renderRanked = (ranked) =>
    renderHostRecord(
      ranked.host,
      priorityMeter(ranked.priority, ranking.maxPriority),
      renderWhy(ranked),
      actions,
    );
  const list = el("ol", { class: "hosts", id: `group-${index}-list` });
  const more = fillCapped(list, hosts, renderRanked, HOSTS_SHOWN, "host");
  return [
    el(
      "p",
      { class: "panel-note" },
      "Highest priority first, and under each host why it ranks where it" +
        " does. Grouped by ",
      el("span", { class: "mono", text: group.key }),
      ".",
    ),
    hostColumnNames(),
    list,
    more,
  ];
}

/**
 * Returns one `HostGroup`: a row that opens onto its hosts. The hosts are
 * rendered when the group is first opened, so a group of hundreds costs
 * nothing until someone asks for it.
 */
function renderGroup(group, index, ranking, actions) {
  const panelId = `group-${index}-hosts`;
  const panel = el("div", { class: "group-panel", id: panelId, hidden: true });
  const toggle = el(
    "button",
    {
      type: "button",
      class: "group-toggle",
      "aria-expanded": "false",
      "aria-controls": panelId,
    },
    el("span", { class: "chevron", "aria-hidden": "true" }),
    el("span", { class: "visually-hidden", text: `Hosts in ${group.label}` }),
  );
  const onClick = () => {
    // The whole row is the target, but a drag that selected text in it (to
    // copy a group name) is not a click on it.
    if (String(window.getSelection()) === "") {
      togglePanel(toggle, panel, () =>
        renderGroupHosts(group, index, ranking, actions),
      );
    }
  };
  const count = (number, unit) => [
    el("span", { class: "group-count", text: NUMBER.format(number) }),
    el("span", { class: "unit", text: ` ${unit}` }),
  ];
  const hostsUnit = group.count === 1 ? "host" : "hosts";
  return el(
    "li",
    { class: "group" },
    el(
      "div",
      { class: "group-head", on: { click: onClick } },
      toggle,
      el("div", { class: "group-name", text: group.label }),
      el("div", { class: "group-hosts num" }, count(group.count, hostsUnit)),
      el(
        "div",
        { class: "group-facing num" },
        count(group.internet_facing_count, "internet-facing"),
      ),
      el(
        "div",
        { class: "group-priority" },
        priorityMeter(group.priority, ranking.maxPriority),
      ),
      el("div", {
        class: "group-examples",
        text: group.example_hosts.join(", "),
      }),
    ),
    panel,
  );
}

/** Returns the `groups` as an ordered list under a row of column names. */
function renderGroups(groups, ranking, actions) {
  const names = columnNames("groups-header", [
    [""],
    ["Group"],
    ["Hosts", "num"],
    ["Internet-facing", "num"],
    ["Priority"],
    ["Example hosts"],
  ]);
  const items = groups.map((group, index) =>
    renderGroup(group, index, ranking, actions),
  );
  const label = "Groups of affected hosts, highest priority first";
  return el(
    "div",
    { class: "groups" },
    names,
    el("ol", { "aria-label": label }, items),
  );
}

/** Returns the collapsed list of affected hosts that are not running. */
function renderInactiveHosts(hosts, actions) {
  const renderInactive = (host) =>
    renderHostRecord(
      host,
      el("span", { class: "muted", text: "not ranked" }),
      el(
        "p",
        { class: "host-detail" },
        "Last scanned ",
        host.last_scan ? timestamp(host.last_scan) : "never",
      ),
      actions,
    );
  const count = plural(hosts.length, "affected host");
  return disclosure({
    id: "inactive-hosts",
    label: "Inactive hosts",
    note: `${count} not running: listed, not ranked`,
    render: () => [
      hostColumnNames(),
      el("ul", { class: "hosts" }, hosts.map(renderInactive)),
    ],
  });
}

/** Returns the section that sizes and orders the affected hosts. */
function renderBlastRadius(response, actions) {
  const { hosts, groups, inactive_hosts: inactive } = response;
  // What a group needs to list its hosts: each `RankedHost` by id, and the
  // priority every bar is drawn against.
  const ranking = {
    hostsById: new Map(hosts.map((ranked) => [ranked.host.id, ranked])),
    maxPriority: [...hosts, ...groups].reduce(
      (max, item) => Math.max(max, item.priority),
      0,
    ),
  };
  return section({
    id: "radius",
    index: "03",
    title: "Blast radius",
    meta: [
      "Resolved from the scanner's detections and ordered by priority =" +
        " threat × exposure.",
      "Bars are relative to the highest priority in this answer.",
    ],
    body: [
      renderHeadline(response),
      groups.length > 0
        ? renderGroups(groups, ranking, actions)
        : el("p", {
            class: "muted",
            text: "No running host carries a matched check.",
          }),
      inactive.length > 0 && renderInactiveHosts(inactive, actions),
    ],
  });
}

// ---------------------------------------------------------------------------
// Report: fix evidence
// ---------------------------------------------------------------------------

/**
 * Returns the evidence under its headings, in `EVIDENCE_KINDS` order. A kind
 * this page has not heard of still shows, last, under its own name.
 *
 * @returns {Array<[string, object[]]>} Heading and items, empty ones left out.
 */
function groupEvidence(items) {
  const headings = new Map(EVIDENCE_KINDS);
  for (const item of items) {
    if (!headings.has(item.kind)) {
      headings.set(item.kind, words(item.kind));
    }
  }
  return [...headings]
    .map(([kind, heading]) => [
      heading,
      items.filter((item) => item.kind === kind),
    ])
    .filter(([, group]) => group.length > 0);
}

/** Returns one `FixEvidence`: its citable id, its document, what it says. */
function renderEvidence(item, sources, actions) {
  const chip = citeChip({ id: item.id, source: sources.get(item.id) }, actions);
  return el(
    "li",
    { class: "evidence" },
    el("span", {}, chip),
    el("span", { class: "doc-ref", text: docLabel(item) }),
    el("p", { class: "evidence-text" }, textWithReference(item.text, item.url)),
  );
}

/** Returns what the data says about a fix, or that it says nothing. */
function renderFixEvidence(response, sources, actions) {
  const groups = groupEvidence(response.fix_evidence).map(([heading, items]) =>
    block(
      "evidence-group",
      heading,
      el(
        "ul",
        { class: "evidence-list" },
        items.map((item) => renderEvidence(item, sources, actions)),
      ),
    ),
  );
  const nothing = el("p", {
    class: "muted",
    text: "The data holds no fix guidance for the matched checks.",
  });
  return section({
    id: "evidence",
    index: "04",
    title: "Fix evidence",
    meta: [
      "Only what the scanner data itself says about a fix. Nothing here is" +
        " written by the language model.",
    ],
    body: groups.length > 0 ? groups : [nothing],
  });
}

// ---------------------------------------------------------------------------
// Report: how it was computed
// ---------------------------------------------------------------------------

/** Returns the total duration of a trace, in milliseconds. */
function traceDuration(steps) {
  return steps.reduce((sum, step) => sum + step.duration_ms, 0);
}

/** Returns the trace as a table whose bars line up into a timeline. */
function renderTrace(steps) {
  const total = traceDuration(steps);
  let elapsed = 0;
  const rows = steps.map((step) => {
    const bar = el("span", { class: "waterfall-bar" });
    const width = shareOf(step.duration_ms, total);
    bar.style.setProperty("--start", shareOf(elapsed, total).toFixed(4));
    bar.style.setProperty("--fill", width.toFixed(4));
    elapsed += step.duration_ms;
    const summary = Object.entries(step.summary).map(([key, value]) => [
      words(key),
      formatValue(value),
    ]);
    return el(
      "tr",
      {},
      el(
        "th",
        { scope: "row" },
        el("span", { class: "step-name", text: step.name }),
      ),
      el(
        "td",
        {},
        formatDuration(step.duration_ms),
        el("div", { class: "waterfall", "aria-hidden": "true" }, bar),
      ),
      el("td", {}, summary.length > 0 && keyValues(summary, "kv-inline")),
    );
  });
  return el(
    "div",
    { class: "table-scroll" },
    el(
      "table",
      { class: "data-table trace-table" },
      tableHead([["Step"], ["Duration"], ["What it produced"]]),
      el("tbody", {}, rows),
    ),
  );
}

/** Returns step 1's reading of the query. */
function renderParsed(parsed) {
  return keyValues([
    ["Query as typed", el("span", { class: "parsed-raw", text: parsed.raw })],
    ["CVE ids", formatValue(parsed.cve_ids)],
    ["QIDs", formatValue(parsed.qids)],
    ["Product", formatValue(parsed.product)],
    ["Version", formatValue(parsed.version)],
    ["Search queries", formatValue(parsed.search_queries)],
    ["Parsed with the language model", formatValue(parsed.used_llm)],
  ]);
}

/**
 * Returns every source the brief was allowed to cite, cited or not, so that
 * what the language model was shown can be read in full.
 */
function renderSources(context, actions) {
  if (context.length === 0) {
    return el("p", { class: "muted", text: "None: nothing matched." });
  }
  const items = context.map((item) =>
    el(
      "li",
      { class: "source-item" },
      el("span", {}, citeChip({ id: item.id, source: item }, actions)),
      el("span", { class: "doc-ref", text: docLabel(item) }),
      el("span", { text: item.title }),
    ),
  );
  return el("ul", { class: "source-list" }, items);
}

/** Returns the `ResponseMeta`: enough to reproduce or audit the answer. */
function renderProvenance(meta) {
  const hashes = (group, label) =>
    Object.entries(group).map(([name, value]) => [
      `${label}: ${name}`,
      hash(value),
    ]);
  const built = meta.artifact_built_at;
  return keyValues([
    ["Version", meta.version],
    ["Index built", built ? timestamp(built) : "unknown"],
    ...hashes(meta.dataset_sha256, "Dataset SHA-256"),
    ["Embedding model", formatValue(meta.embedding_model)],
    ["Reranker", formatValue(meta.rerank_model)],
    ["Language model provider", formatValue(meta.llm_provider)],
    ["Language model", formatValue(meta.llm_model)],
    ...hashes(meta.prompt_sha256, "Prompt SHA-256"),
  ]);
}

/** Returns the collapsed audit section; `index` numbers it in the rail. */
function renderAudit(response, index, actions) {
  const { trace, parsed, context, meta } = response;
  const sourcesTitle = `Citable sources (${NUMBER.format(context.length)})`;
  const steps = plural(trace.length, "step");
  return section({
    id: "audit",
    index,
    title: "How this was computed",
    body: [
      disclosure({
        id: "audit-detail",
        label: "Trace, parsed query, sources and provenance",
        note: `${steps} in ${formatDuration(traceDuration(trace))}`,
        render: () => [
          block("audit-block", "Trace", renderTrace(trace)),
          block("audit-block", "Parsed query", renderParsed(parsed)),
          block("audit-block", sourcesTitle, renderSources(context, actions)),
          block("audit-block", "Provenance", renderProvenance(meta)),
        ],
      }),
    ],
  });
}

/** Returns every section of a response, in the order an analyst reads them. */
function renderReport(response, actions) {
  const sources = new Map(response.context.map((item) => [item.id, item]));
  const sections =
    response.status === "no_match"
      ? [renderNoMatch(response), renderAudit(response, "02", actions)]
      : [
          renderAnswer(response, sources, actions),
          renderMatches(response, sources, actions),
          renderBlastRadius(response, actions),
          renderFixEvidence(response, sources, actions),
          renderAudit(response, "05", actions),
        ];
  // Sections arrive one after the other; see "Motion" in the stylesheet.
  sections.forEach((node, order) =>
    node.style.setProperty("--order", String(order)),
  );
  return sections;
}

// ---------------------------------------------------------------------------
// Drawer contents
// ---------------------------------------------------------------------------

/**
 * Returns the drawer view of one source, with `quote` highlighted in it.
 *
 * @param {object} source A `ContextItem`.
 * @param {string|null} quote The cited text, or null to just show the source.
 * @returns {{kicker: string, title: string, body: Node[]}}
 */
function renderSource(source, quote) {
  const span = quote === null ? null : locateQuote(source.text, quote);
  const kind = DOC_TYPE_LABELS[source.doc_type] ?? source.doc_type;
  const quoted =
    quote !== null &&
    block(
      "drawer-section",
      "Quoted",
      el("blockquote", {
        class: span ? "quote-block" : "quote-block is-missing",
        text: quote,
      }),
      el("p", {
        class: "quote-status",
        "data-found": String(span !== null),
        text: span
          ? "Found in this source, highlighted below."
          : "Quote not found in this source.",
      }),
    );
  const text = el(
    "p",
    { class: "source-text" },
    highlighted(source.text, span),
  );
  return {
    kicker: `Source ${source.id} · ${kind}`,
    title: docLabel(source),
    body: [
      source.title && el("p", { class: "source-title", text: source.title }),
      quoted,
      block("drawer-section", "Text of the source", text),
    ].filter(Boolean),
  };
}

/** Returns a host's descriptive fields as a definition list. */
function renderHostFacts(host) {
  const optional = (value) => value ?? "none";
  return keyValues([
    ["State", host.state.toLowerCase()],
    ["Criticality", `${host.criticality} of 5`],
    ["Exposure", host.internet_facing ? "internet-facing" : "internal"],
    ["Public IP", optional(host.public_ip)],
    ["Private IP", optional(host.private_ip)],
    ["Operating system", optional(host.os)],
    ["Region", optional(host.region)],
    ["VPC", optional(host.vpc_id)],
    ["Security group", optional(host.security_group)],
    ["Cluster", optional(host.cluster)],
    ["Role", optional(host.role)],
    ["Container runtime", host.is_docker_host ? "yes" : "no"],
    ["Last scan", host.last_scan ? timestamp(host.last_scan) : "never"],
    ["Host id", host.id],
  ]);
}

/** Returns the open ports as a table, or a sentence when there are none. */
function renderPorts(ports) {
  if (ports.length === 0) {
    return el("p", { class: "muted", text: "The scanner found no open port." });
  }
  const rows = ports.map((port) =>
    el(
      "tr",
      {},
      el("th", { scope: "row", class: "mono", text: port.port }),
      el("td", { text: port.protocol }),
      el("td", { text: port.service ?? "not identified" }),
    ),
  );
  return el(
    "table",
    { class: "data-table" },
    tableHead([["Port"], ["Protocol"], ["Service"]]),
    el("tbody", {}, rows),
  );
}

/**
 * Returns detections as a table.
 *
 * @param {object[]} detections `Detection`s to list.
 * @param {object} labels `HostDetail.qid_labels`; an explained QID has one.
 * @param {Set<string>} matchedQids QIDs the response on screen matched.
 */
function renderDetections(detections, labels, matchedQids) {
  const when = (iso) => (iso ? timestamp(iso, DATE) : "unknown");
  const rows = detections.map((detection) =>
    el(
      "tr",
      {},
      el(
        "th",
        { scope: "row" },
        el("span", { class: "mono", text: `QID ${detection.qid}` }),
        labels[detection.qid] && el("div", { text: labels[detection.qid] }),
        matchedQids.has(detection.qid) &&
          badge("matched by this query", "accent"),
      ),
      el("td", { class: "nowrap" }, when(detection.first_found)),
      el("td", { class: "nowrap" }, when(detection.last_found)),
    ),
  );
  return el(
    "div",
    { class: "table-scroll" },
    el(
      "table",
      { class: "data-table" },
      tableHead([["Check"], ["First found"], ["Last found"]]),
      el("tbody", {}, rows),
    ),
  );
}

/**
 * Returns the drawer body for a `HostDetail`. Detections the corpus can
 * explain and those it cannot are kept apart, because only the first kind
 * can ever match a query.
 */
function renderHostDetail(detail, matchedQids) {
  const { host, ports, detections, qid_labels: labels } = detail;
  const explained = detections.filter((detection) => detection.explained);
  const unexplained = detections.filter((detection) => !detection.explained);
  const bareQids = new Set(unexplained.map((detection) => detection.qid));
  const title = (name, items) => `${name} (${NUMBER.format(items.length)})`;
  return [
    renderHostFacts(host),
    block("drawer-section", title("Open ports", ports), renderPorts(ports)),
    block(
      "drawer-section",
      title("Explained detections", explained),
      explained.length > 0
        ? renderDetections(explained, labels, matchedQids)
        : el("p", {
            class: "muted",
            text: "None of this host's detections has a write-up.",
          }),
    ),
    block(
      "drawer-section",
      title("Unexplained detections", unexplained),
      el("p", {
        class: "muted",
        text:
          "The data gives only a number for these checks, so no query can" +
          " match them and nothing here says what they mean.",
      }),
      unexplained.length > 0 &&
        disclosure({
          id: "unexplained-detections",
          label: "List them",
          note: plural(bareQids.size, "QID"),
          render: () => renderDetections(unexplained, labels, matchedQids),
        }),
    ),
  ];
}

// ---------------------------------------------------------------------------
// States around a report
// ---------------------------------------------------------------------------

/** Rewrites the header's status pill from a `Readiness`. */
function renderStatus(pill, readiness) {
  const part = (...content) => el("span", { class: "status-part" }, content);
  const states = {
    ready: "Ready",
    not_ready: "Not ready",
    unreachable: "Service unreachable",
  };
  const parts = [];
  if (readiness.status === "ready") {
    const meta = readiness.meta ?? {};
    const model = el("span", { class: "mono", text: meta.llm_model });
    parts.push(
      meta.llm_model
        ? part("brief by ", model)
        : part("no language model: answers come without a written brief"),
    );
    if (meta.artifact_built_at) {
      parts.push(part("index built ", timestamp(meta.artifact_built_at)));
    }
  } else if (readiness.reason) {
    parts.push(part(readiness.reason));
  }
  pill.dataset.state = readiness.status;
  pill.replaceChildren(
    el("span", { class: "status-dot", "aria-hidden": "true" }),
    el("span", { class: "status-state", text: states[readiness.status] }),
    ...parts,
  );
}

/** Returns what the loaded index covers, from `CorpusStats`. */
function renderIndexStats(stats) {
  const n = (value) => NUMBER.format(value);
  const share = stats.total_detections
    ? Math.round((100 * stats.explained_detections) / stats.total_detections)
    : 0;
  return section({
    id: "index",
    title: "Loaded index",
    body: [
      statRow([
        {
          value: n(stats.hosts),
          label: `hosts, ${n(stats.hosts_with_detections)} with detections`,
        },
        {
          value: `${n(stats.explained_qids)} of ${n(stats.total_qids)}`,
          label: "detected QIDs have a write-up",
        },
        {
          value: `${share}%`,
          label: `of ${n(stats.total_detections)} detections can be explained`,
        },
        {
          value: n(stats.cves),
          label: `CVEs, in ${n(stats.chunks)} searchable chunks`,
        },
      ]),
      el("p", {
        class: "index-note",
        text:
          "Only checks with a write-up can match a query. For the rest the" +
          " data holds a number and nothing else, so no match is never" +
          " reported as not affected.",
      }),
    ],
  });
}

/** Returns the section that says the service cannot take a query yet. */
function renderNotReady(readiness) {
  const waiting = readiness.status === "not_ready";
  return section({
    id: "not-ready",
    title: waiting ? "Not ready" : "Unreachable",
    body: [
      callout(
        "waiting",
        waiting ? FAILURE_TITLES.not_ready : FAILURE_TITLES.network,
        el("p", { text: readiness.reason ?? "The service gave no reason." }),
        el("p", {
          class: "muted",
          text: "This page asks again every few seconds.",
        }),
      ),
    ],
  });
}

/**
 * Returns the page before the first query: how to ask, and what the service
 * has loaded or why it cannot answer yet. `readiness` is null until the
 * first check returns.
 */
function renderStart(template, readiness) {
  const ways = fromTemplate(template);
  if (readiness === null) {
    return ways;
  }
  if (readiness.status !== "ready") {
    return [renderNotReady(readiness), ...ways];
  }
  return readiness.stats ? [...ways, renderIndexStats(readiness.stats)] : ways;
}

/** Returns a failure as a callout, with the request id and a way forward. */
function renderFailure(failure, onRetry) {
  const hint = FAILURE_HINTS[failure.kind];
  return callout(
    "error",
    failure.message,
    failure.detail && el("p", { class: "error-detail", text: failure.detail }),
    hint && el("p", { class: "error-detail", text: hint }),
    failure.requestId &&
      el(
        "p",
        { class: "request-id" },
        "Request id ",
        el("span", { class: "mono", text: failure.requestId }),
      ),
    onRetry &&
      el("button", {
        type: "button",
        class: "button button-quiet",
        text: "Try again",
        on: { click: onRetry },
      }),
  );
}

// ---------------------------------------------------------------------------
// Controllers
// ---------------------------------------------------------------------------

/**
 * Wires the drawer: a side panel (a bottom sheet on narrow screens) that is
 * not modal, so an analyst can keep it open and walk from one citation to
 * the next. It remembers only which control opened it, to mark that control
 * as expanded and to hand focus back on close.
 *
 * @returns {{open: function, close: function}}
 */
function createDrawer() {
  const root = document.getElementById("drawer");
  const kicker = document.getElementById("drawer-kicker");
  const title = document.getElementById("drawer-title");
  const body = document.getElementById("drawer-body");
  let opener = null;
  let pending = new AbortController();

  function release() {
    pending.abort();
    opener?.setAttribute("aria-expanded", "false");
  }

  /**
   * Shows a view, replacing whatever was open.
   *
   * @param {{kicker: string, title: string, body: Node[]}} view
   * @param {HTMLElement} from The control that asked.
   * @returns {{signal: AbortSignal, fill: function(Node[]): void}} For
   *   content that arrives later: `signal` fires, and `fill` stops working,
   *   once the drawer has moved on.
   */
  function open(view, from) {
    release();
    pending = new AbortController();
    opener = from;
    opener.setAttribute("aria-expanded", "true");
    kicker.textContent = view.kicker;
    title.textContent = view.title;
    const { signal } = pending;
    const fill = (nodes) => {
      if (signal.aborted) {
        return;
      }
      body.replaceChildren(...nodes);
      // Start at the highlight when there is one, a little below the top.
      const mark = body.querySelector("mark");
      const top = mark ? mark.offsetTop - body.clientHeight / 3 : 0;
      body.scrollTop = Math.max(top, 0);
    };
    root.hidden = false;
    fill(view.body);
    title.focus({ preventScroll: true });
    return { signal, fill };
  }

  function close() {
    if (root.hidden) {
      return;
    }
    release();
    root.hidden = true;
    body.replaceChildren();
    if (opener?.isConnected) {
      opener.focus();
    }
    opener = null;
  }

  document.getElementById("drawer-close").addEventListener("click", close);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !root.hidden) {
      event.preventDefault();
      close();
    }
  });
  return { open, close };
}

/** Opens the drawer on a host and fills it in when `HostDetail` arrives. */
async function showHost(drawer, host, matchedQids, from) {
  const loading = el("p", {
    class: "loading-note",
    role: "status",
    text: "Loading ports and detections…",
  });
  const view = drawer.open(
    { kicker: "Host", title: host.name, body: [loading] },
    from,
  );
  try {
    const detail = await fetchHost(host.id, view.signal);
    view.fill(renderHostDetail(detail, matchedQids));
  } catch (error) {
    if (error.name !== "AbortError") {
      const retry = () => showHost(drawer, host, matchedQids, from);
      view.fill([renderFailure(asFailure(error), retry)]);
    }
  }
}

/**
 * Returns a function that asks `/readyz` until the service is ready,
 * reporting every answer to `onAnswer`. Calling it while it is already
 * asking does nothing, so it is safe to call again after a 503.
 */
function createReadinessWatcher(onAnswer) {
  let watching = false;
  return async function watch() {
    if (watching) {
      return;
    }
    watching = true;
    try {
      for (;;) {
        const readiness = await fetchReadiness();
        onAnswer(readiness);
        if (readiness.status === "ready") {
          return;
        }
        await new Promise((resolve) => {
          setTimeout(resolve, READINESS_RETRY_MS);
        });
      }
    } finally {
      watching = false;
    }
  };
}

/** Returns why `query` cannot be sent, or null when it can. */
function queryProblem(query) {
  const length = [...query].length;
  if (length === 0) {
    return "Enter a CVE id, a QID or the text of an advisory.";
  }
  if (length > MAX_QUERY_CHARS) {
    return (
      `The service accepts ${NUMBER.format(MAX_QUERY_CHARS)} characters;` +
      ` this is ${NUMBER.format(length)}.`
    );
  }
  return null;
}

/** Returns what a screen reader is told when a response arrives. */
function announcementFor(response) {
  if (response.status === "no_match") {
    return "Analysis complete. Nothing in the corpus matches this query.";
  }
  const hosts = plural(response.hosts.length, "running affected host");
  const groups = plural(response.groups.length, "group");
  return `Analysis complete. ${hosts} in ${groups}.`;
}

/** Wires the page. */
function main() {
  const form = document.getElementById("ask-form");
  const input = document.getElementById("query");
  const feedback = document.getElementById("query-feedback");
  const submit = document.getElementById("analyse");
  const cancel = document.getElementById("cancel");
  const chips = [...document.querySelectorAll(".example-chip")];
  const results = document.getElementById("results");
  const announcer = document.getElementById("announcer");
  const pill = document.getElementById("service-status");
  const emptyTemplate = document.getElementById("empty-template");
  const skeletonTemplate = document.getElementById("skeleton-template");
  const drawer = createDrawer();

  // The page's whole state: what the service last said about itself, the
  // response on screen, and the request in flight. Which view `results`
  // holds is recorded on the element itself, as `data-view`.
  let readiness = null;
  let current = null;
  let inflight = null;

  const actions = {
    openSource: (source, quote, from) =>
      drawer.open(renderSource(source, quote), from),
    openHost: (host, from) => {
      const matchedQids = new Set(current.matches.map((match) => match.qid));
      showHost(drawer, host, matchedQids, from);
    },
  };

  function show(view, nodes) {
    results.dataset.view = view;
    results.replaceChildren(...nodes);
  }

  function showStart() {
    show("start", renderStart(emptyTemplate, readiness));
  }

  function showFailure(failure, onRetry) {
    const body = [renderFailure(failure, onRetry)];
    show("failure", [section({ id: "failure", title: "Error", body })]);
  }

  function showReport(response) {
    try {
      show("report", renderReport(response, actions));
    } catch (error) {
      showFailure(asFailure(error), null);
    }
  }

  function setFeedback(text, tone = "") {
    feedback.textContent = text;
    feedback.dataset.tone = tone;
    input.setAttribute("aria-invalid", String(tone === "error"));
  }

  function updateCounter() {
    // UTF-16 length is never below the character count, so it is a cheap
    // test for "nowhere near the limit".
    if (input.value.length < COUNTER_FROM_CHARS) {
      setFeedback("");
      return;
    }
    const length = [...input.value.trim()].length;
    const limit = NUMBER.format(MAX_QUERY_CHARS);
    setFeedback(
      `${NUMBER.format(length)} of ${limit} characters`,
      length > MAX_QUERY_CHARS ? "error" : "",
    );
  }

  function setBusy(busy) {
    // `aria-disabled`, not `disabled`: a disabled button drops the focus it
    // holds, and the keyboard user with it.
    submit.setAttribute("aria-disabled", String(busy));
    submit.querySelector(".button-label").textContent = busy
      ? "Analysing…"
      : "Analyse";
    chips.forEach((chip) => chip.setAttribute("aria-disabled", String(busy)));
    cancel.hidden = !busy;
  }

  function bringResultsIntoView() {
    if (results.getBoundingClientRect().top > window.innerHeight * 0.6) {
      const calm = window.matchMedia("(prefers-reduced-motion: reduce)");
      results.scrollIntoView({
        block: "start",
        behavior: calm.matches ? "auto" : "smooth",
      });
    }
  }

  const watchReadiness = createReadinessWatcher((answer) => {
    // The watcher reports every answer. Only a changed one is repainted, or
    // a screen reader would hear "Not ready" every few seconds.
    if (JSON.stringify(answer) === JSON.stringify(readiness)) {
      return;
    }
    readiness = answer;
    renderStatus(pill, answer);
    if (results.dataset.view === "start") {
      showStart();
    }
  });

  async function run(query) {
    const request = new AbortController();
    inflight = request;
    drawer.close();
    setBusy(true);
    show("loading", fromTemplate(skeletonTemplate));
    announcer.textContent = "Analysing.";
    try {
      current = await analyse(query, request.signal);
      showReport(current);
      announcer.textContent = announcementFor(current);
      bringResultsIntoView();
    } catch (error) {
      if (error.name === "AbortError") {
        announcer.textContent = "Cancelled.";
        if (current) {
          showReport(current);
        } else {
          showStart();
        }
        return;
      }
      const failure = asFailure(error);
      showFailure(failure, () => run(query));
      announcer.textContent = failure.message;
      if (failure.kind === "not_ready") {
        watchReadiness();
      }
    } finally {
      inflight = null;
      setBusy(false);
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (inflight) {
      return;
    }
    const query = input.value.trim();
    const problem = queryProblem(query);
    if (problem) {
      setFeedback(problem, "error");
      input.focus();
      return;
    }
    run(query);
  });

  input.addEventListener("input", updateCounter);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  cancel.addEventListener("click", () => inflight?.abort());

  for (const chip of chips) {
    chip.addEventListener("click", () => {
      if (inflight) {
        return;
      }
      input.value = chip.dataset.example;
      updateCounter();
      form.requestSubmit();
    });
  }

  const platform = navigator.userAgentData?.platform ?? navigator.platform;
  if (/mac|iphone|ipad/i.test(platform)) {
    document.getElementById("submit-modifier").textContent = "⌘";
  }

  showStart();
  watchReadiness();
}

main();
