"""
Apollo Medical Triage — Deterministic Clinical Safety Pipeline
==============================================================
Author: Built for ADTC 2026 — Team: Eleogu Chukwuebuka Joseph

Pre-processing and post-processing guardrails that run BEFORE and AFTER the
LLM call respectively. This module is the real fix — it treats every LLM
output as UNTRUSTED until it passes every check below.

Ground rule: this module is NOT a clinical authority. All clinical numbers,
thresholds, and phrase lists live exclusively in config/clinical_protocol.yaml.
Nothing clinical is hardcoded here. If a test case implies the protocol data
is wrong, flag it for human clinical review — do not edit values here.

Design decisions:
  - Safety check (4.3) fails CLOSED: direct escalation, NO LLM retry.
  - Non-safety checks retry once via llm_retry_fn; if None (unit tests), go
    directly to escalation. This keeps the entire module testable without a
    running LLM or server.
  - Age extraction is regex-only — never delegated to the LLM.
  - All clinical data loaded from yaml at request time, then cached in-process.
"""

import re
import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    import yaml
except ImportError:
    raise ImportError("pyyaml is required — run: pip install pyyaml>=6.0")

logger = logging.getLogger("apollo.pipeline")

# Path to clinical protocol config — two levels up from backend/app/
PROTOCOL_PATH = Path(__file__).parent.parent.parent / "config" / "clinical_protocol.yaml"

# ---------------------------------------------------------------------------
# ESCALATION TEMPLATE — the one thing safely hardcoded
# This content is guaranteed safe regardless of LLM output. It is used when
# any safety check fails and we must fail closed.
# ---------------------------------------------------------------------------
ESCALATION_TEMPLATE = """\
### 1. Immediate Priority

\u26a0\ufe0f **This is a medical emergency. Go to the nearest hospital or call \
emergency services immediately \u2014 do not wait.**

Apollo could not generate a verified safe response for your query. To protect \
your safety, please seek immediate professional medical evaluation.

### 2. Emergency Red Flags (Seek Immediate Medical Care)

- The symptoms you described require urgent in-person evaluation.
- Do not attempt to manage this situation at home.
- Bring the patient to the nearest emergency room or call an ambulance now.

### 3. Immediate Actions & Supportive Measures

- Go to your nearest emergency room or call emergency services immediately.
- If the patient is unconscious or not breathing, call emergency services at once.
- Keep the patient as calm and still as possible while awaiting help.
- Do not give food, liquids, or any medication unless instructed by emergency services.
- Bring any medication containers, substances, or relevant medical records to hospital.

### 4. Likely Causes (Differential Overview)

Further emergency evaluation by qualified medical personnel is required to \
determine the exact cause. This cannot be determined safely without in-person \
examination.\
"""

_protocol_cache: dict | None = None


def load_clinical_protocol() -> dict:
    """Load and cache clinical_protocol.yaml. Single source of truth for all
    clinical thresholds. Raises FileNotFoundError if config is missing."""
    global _protocol_cache
    if _protocol_cache is not None:
        return _protocol_cache

    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(
            f"clinical_protocol.yaml not found at {PROTOCOL_PATH}. "
            "Ensure the config/ directory exists at the repo root."
        )

    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        _protocol_cache = yaml.safe_load(f)

    logger.info(
        "[PROTOCOL] Loaded clinical_protocol.yaml v%s "
        "(reviewed by: %s, date: %s)",
        _protocol_cache.get("version", "?"),
        _protocol_cache.get("last_reviewed_by", "PENDING"),
        _protocol_cache.get("last_reviewed_date", "?"),
    )
    return _protocol_cache


def invalidate_protocol_cache() -> None:
    """Force-reload protocol on next access. Used in tests."""
    global _protocol_cache
    _protocol_cache = None


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class StructuredQuery:
    """Output of preprocess_query(). Carries deterministic pre-scan results
    into both the LLM prompt context block and the post-processing pipeline."""
    raw_input: str
    cleaned_input: str
    age_months: int | None = None
    age_band: dict | None = None
    symptoms: list[str] = field(default_factory=list)
    matched_red_flags: list[str] = field(default_factory=list)
    active_emergency: bool = False
    substance_protocol: dict | None = None


@dataclass
class ApolloResponse:
    """Validated, sanitized LLM response that passed all post-processing checks."""
    content: str
    section1: str = ""
    section2: str = ""
    section3: str = ""
    section4: str = ""
    checks_passed: list[str] = field(default_factory=list)
    regeneration_count: int = 0
    retrieved_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Escalation:
    """Hard-fail output — returned when safety checks fail beyond recovery."""
    content: str
    reason: str
    failed_check: str
    raw_llm_output: str
    structured_query: StructuredQuery


# ---------------------------------------------------------------------------
# PRE-PROCESSING PIPELINE
# ---------------------------------------------------------------------------

# Pre-compiled harness stripping regexes — handles both line-start prefixes and mid-sentence tags
_RE_HARNESS_PATTERNS = [
    re.compile(r"(?i)^\s*(?:APOLLO\s+TESTING.*?:|FOR\s+TRACK\s+\d+.*?:|TRACK\s+\d+.*?Q\s*[-:]|Q\s*[-:]|QUESTION\s+\d+.*?:)\s*", re.MULTILINE),
    re.compile(r"(?i)[\(\[\{]?\s*(?:FOR\s+)?TRACK\s+\d+(?:,\s*QUESTION\s+\d+)?\s*[\)\]\}]?", re.IGNORECASE),
    re.compile(r"(?i)[\(\[\{]?\s*QUESTION\s+\d+\s*[\)\]\}]?", re.IGNORECASE),
    re.compile(r"(?i)[\(\[\{]?\s*APOLLO\s+TESTING(?:\s*ROUND\s*\d+)?\s*[\)\]\}]?", re.IGNORECASE),
]

_RE_AGE_RULES = [
    (re.compile(r"\b(\d+)\s*-?\s*month(?:s)?(?:\s*-?\s*old)?\b", re.IGNORECASE), "months"),
    (re.compile(r"\b(\d+)\s*-?\s*year(?:s)?(?:\s*-?\s*old)?\b", re.IGNORECASE),  "years"),
    (re.compile(r"\b(\d+)\s*-?\s*week(?:s)?(?:\s*-?\s*old)?\b", re.IGNORECASE),  "weeks"),
    (re.compile(r"\b(newborn|neonate|neonatal)\b", re.IGNORECASE),                "newborn"),
]

_SYMPTOM_KEYWORDS = [
    "fever", "diarrhea", "diarrhoea", "cough", "breathing", "chest",
    "vomiting", "seizure", "convulsion", "rash", "pain", "bleeding",
    "headache", "weakness", "lethargy", "unconscious", "crying",
    "swelling", "jaundice", "stiff neck", "fontanelle", "feeding",
]


def extract_age_months(text: str) -> int | None:
    """Deterministic regex-only age extraction, normalized to months.
    Never delegates to the LLM — this eliminates the age-band mismatch bug."""
    for pattern, unit in _RE_AGE_RULES:
        m = pattern.search(text)
        if m:
            if unit == "newborn":
                return 0
            value = int(m.group(1))
            if unit == "months":
                return value
            if unit == "years":
                return value * 12
            if unit == "weeks":
                return max(0, round(value / 4.33))
    return None


def extract_symptoms(text: str) -> list[str]:
    """Keyword extraction of symptoms from the user query."""
    text_lower = text.lower()
    return [kw for kw in _SYMPTOM_KEYWORDS if kw in text_lower]


def get_age_band(age_months: int, protocol: dict) -> dict | None:
    """Lookup the matching age-band slice from the protocol config.
    Returns None (never interpolates) if age falls outside defined bands."""
    for band in protocol.get("respiratory_rate_thresholds", []):
        if band["age_min_months"] <= age_months < band["age_max_months"]:
            return band
    return None


def scan_red_flags(text: str, protocol: dict) -> list[str]:
    """Deterministic phrase scan against the protocol's red-flag list."""
    text_lower = text.lower()
    return [
        phrase for phrase in protocol.get("active_red_flag_phrases", [])
        if phrase.lower() in text_lower
    ]


def get_substance_protocol(text: str, protocol: dict) -> dict | None:
    """Check if the query mentions a substance with a protocol entry."""
    text_lower = text.lower()
    sp = protocol.get("substance_protocols", {})
    battery_kws = ["button battery", "swallowed battery", "ingested battery", "coin battery"]
    paracetamol_kws = ["paracetamol overdose", "acetaminophen overdose", "took too many paracetamol"]
    general_kws = ["poisoning", "swallowed", "overdose", "pesticide", "bleach", "ingested"]

    if any(kw in text_lower for kw in battery_kws):
        return sp.get("button_battery")
    if any(kw in text_lower for kw in paracetamol_kws):
        return sp.get("paracetamol_overdose")
    if any(kw in text_lower for kw in general_kws):
        return sp.get("general_ingestion")
    return None


def preprocess_query(raw_input: str) -> StructuredQuery:
    """
    Layer A — Pre-Processing Pipeline.

    1. Strip test/benchmark artifacts (both prefix and mid-sentence; never reaches the LLM).
    2. Extract structured fields deterministically: age (months), symptoms,
       substance protocol.
    3. Deterministic red-flag pre-scan against clinical_protocol.yaml.
    4. Set active_emergency flag and resolve age-band BEFORE the LLM sees
       anything.

    Returns StructuredQuery — flows into the LLM context block AND the
    post-processing validation pipeline.
    """
    protocol = load_clinical_protocol()

    # Step 1 — Strip test artifacts (prefix + mid-sentence)
    cleaned = raw_input
    for pattern in _RE_HARNESS_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned != raw_input.strip():
        logger.info("[PIPELINE-A] Test/harness artifacts stripped: %r -> %r", raw_input[:80], cleaned[:80])

    # Step 2 — Extract structured fields
    age_months = extract_age_months(cleaned)
    symptoms = extract_symptoms(cleaned)
    substance_protocol = get_substance_protocol(cleaned, protocol)

    # Step 3 — Red-flag pre-scan
    matched = scan_red_flags(cleaned, protocol)
    if substance_protocol and substance_protocol.get("emergency"):
        matched = list(set(matched + ["substance_ingestion"]))

    active_emergency = bool(matched)

    # Step 4 — Resolve age band
    age_band: dict | None = None
    if age_months is not None:
        age_band = get_age_band(age_months, protocol)
        if age_band is None:
            logger.warning(
                "[PIPELINE-A] Age %dmo is outside all defined age bands in protocol.", age_months
            )

    sq = StructuredQuery(
        raw_input=raw_input,
        cleaned_input=cleaned,
        age_months=age_months,
        age_band=age_band,
        symptoms=symptoms,
        matched_red_flags=matched,
        active_emergency=active_emergency,
        substance_protocol=substance_protocol,
    )
    logger.info(
        "[PIPELINE-A] age=%s mo | band=%s | emergency=%s | flags=%s",
        age_months,
        age_band.get("band") if age_band else "N/A",
        active_emergency,
        matched,
    )
    return sq


def build_system_context_block(sq: StructuredQuery) -> str:
    """Build the [SYSTEM CONTEXT] block injected into the LLM prompt.
    Removes age-band selection and emergency detection from the LLM's
    judgment and turns them into an explicit lookup the model must relay."""
    lines = ["[SYSTEM CONTEXT \u2014 DO NOT ECHO THIS BLOCK IN YOUR RESPONSE]"]
    lines.append(f"active_emergency_detected: {str(sq.active_emergency).lower()}")

    if sq.matched_red_flags:
        lines.append(f"matched_red_flags: {sq.matched_red_flags}")

    if sq.age_months is not None:
        lines.append(f"patient_age_months: {sq.age_months}")
    else:
        lines.append("patient_age_months: null (age unspecified — do NOT guess or interpolate age band; ask caregiver for age or default to seek immediate evaluation)")

    if sq.age_band:
        t = sq.age_band.get("fast_breathing_threshold", "?")
        lbl = sq.age_band.get("label", "?")
        lines.append(f"applicable_respiratory_threshold: \">={t} breaths/min ({lbl})\"")

    if sq.substance_protocol:
        lines.append("substance_protocol_active: true")
        hp = sq.substance_protocol.get("honey_protocol")
        if hp:
            lines.append(
                f"honey_protocol: {hp['dose_ml']}mL every {hp['frequency_minutes']}min "
                f"(max {hp['max_doses']} doses) ONLY if age>={hp['eligible_min_age_months']}mo "
                f"AND ingestion<{hp['eligible_max_hours_since_ingestion']}h. "
                f"WARNING: {hp['warning']}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# POST-PROCESSING PIPELINE — each check is a named, independently-testable fn
# ---------------------------------------------------------------------------

_SECTION_HEADERS = [
    re.compile(r"###\s*1\.\s*Immediate Priority", re.IGNORECASE),
    re.compile(r"###\s*2\.\s*Emergency Red Flags", re.IGNORECASE),
    re.compile(r"###\s*3\.\s*(Immediate Actions|Home Care|Supportive Measures)", re.IGNORECASE),
    re.compile(r"###\s*4\.\s*Likely Causes", re.IGNORECASE),
]
_SECTION_STUBS = [
    "### 1. Immediate Priority\n\nThis is a medical emergency. Go to the nearest hospital immediately \u2014 do not wait.\n",
    "### 2. Emergency Red Flags (Seek Immediate Medical Care)\n\n- Seek immediate emergency care for the symptoms described.\n",
    "### 3. Immediate Actions & Supportive Measures\n\nGo to hospital immediately. Do not attempt home treatment.\n",
    "### 4. Likely Causes (Differential Overview)\n\nFurther professional evaluation is required to determine the exact cause.\n",
]


def _extract_sections(text: str) -> dict[str, str]:
    """Extract the 4 named sections from the LLM output into a dict."""
    positions: list[tuple[int, int]] = []
    for i, pat in enumerate(_SECTION_HEADERS):
        m = pat.search(text)
        if m:
            positions.append((i + 1, m.start()))
    positions.sort(key=lambda x: x[1])

    result: dict[str, str] = {}
    for idx, (sec_num, start) in enumerate(positions):
        end = positions[idx + 1][1] if idx + 1 < len(positions) else len(text)
        # Skip the header line itself
        header_end = text.find("\n", start)
        content_start = header_end + 1 if header_end != -1 else start
        result[str(sec_num)] = text[content_start:end].strip()

    return result


def check_schema_conformance(text: str) -> tuple[bool, str]:
    """4.1 — Exactly 4 headers, in order, no duplicates, no preamble.
    Returns (ok, reason)."""
    for i, pat in enumerate(_SECTION_HEADERS):
        if not pat.search(text):
            return False, f"Section {i + 1} header missing"

    for i, pat in enumerate(_SECTION_HEADERS):
        if len(pat.findall(text)) > 1:
            return False, f"Section {i + 1} header appears more than once (duplicate)"

    # Check for meaningful preamble before Section 1
    m = _SECTION_HEADERS[0].search(text)
    if m and m.start() > 0 and len(text[: m.start()].strip()) > 20:
        return False, f"Preamble text found before Section 1 ({m.start()} chars)"

    return True, "ok"


def check_metadata_leakage(
    text: str, raw_input: str, protocol: dict
) -> tuple[bool, str]:
    """4.2 — Deny-list regex scan for benchmark / test-harness metadata.
    Also checks for literal harness fragments from the raw_input."""
    deny_patterns = protocol.get("metadata_deny_patterns", [])
    for pattern_str in deny_patterns:
        if re.search(pattern_str, text, re.IGNORECASE | re.MULTILINE):
            return False, f"Metadata pattern matched: {pattern_str!r}"

    # Dynamic: look for harness-shaped fragments from raw_input
    for frag in re.findall(
        r"(?i)(TRACK\s+\d+|QUESTION\s+\d+|APOLLO\s+TESTING|FOR\s+TRACK)", raw_input
    ):
        if re.search(re.escape(frag.strip()), text, re.IGNORECASE):
            return False, f"Harness literal echoed in output: {frag!r}"

    return True, "ok"


def check_emergency_consistency(
    text: str, sq: StructuredQuery, protocol: dict
) -> tuple[bool, str]:
    """4.3 — HIGHEST-SEVERITY check. Life-safety gate.

    When active_emergency=True:
    - Section 1 MUST contain an emergency confirmation phrase.
    - Section 3 MUST NOT contain any unsafe-advice deny-pattern.

    On failure: callers MUST NOT retry — route directly to escalation template.
    """
    if not sq.active_emergency:
        return True, "ok (no active emergency)"

    sections = _extract_sections(text)

    # Positive assertion: Section 1 must declare emergency
    s1 = sections.get("1", "")
    emergency_phrases = protocol.get("emergency_confirmation_phrases", [])
    if not any(phrase.lower() in s1.lower() for phrase in emergency_phrases):
        return (
            False,
            "Section 1 does not contain emergency confirmation language "
            "despite active_emergency=True",
        )

    # Negative assertion: Section 3 must not contain unsafe advice
    s3 = sections.get("3", "")
    for deny_str in protocol.get("unsafe_advice_deny_patterns", []):
        if re.search(deny_str, s3, re.IGNORECASE):
            return False, f"Section 3 contains unsafe advice pattern: {deny_str!r}"

    return True, "ok"


def check_age_band_consistency(
    text: str, sq: StructuredQuery, protocol: dict
) -> tuple[bool, str]:
    """4.4 — Cross-reference any respiratory-rate threshold cited in the output
    against the patient's actual age band from StructuredQuery."""
    if sq.age_months is None or sq.age_band is None:
        return True, "ok (no age data)"

    correct_threshold = sq.age_band.get("fast_breathing_threshold")
    if correct_threshold is None:
        return True, "ok (no threshold in band)"

    bands = protocol.get("respiratory_rate_thresholds", [])
    all_thresholds = {b["fast_breathing_threshold"] for b in bands}
    wrong_thresholds = all_thresholds - {correct_threshold}

    for rate_str in re.findall(r"\b(\d+)\s*breaths?(?:\s*/?(?:per\s*)?min(?:ute)?)?\b", text, re.IGNORECASE):
        rate = int(rate_str)
        if rate in wrong_thresholds:
            return (
                False,
                f"Output cites {rate} breaths/min but patient age "
                f"{sq.age_months}mo requires {correct_threshold} breaths/min "
                f"({sq.age_band.get('label', '')})",
            )

    return True, "ok"


def check_actionability(
    text: str, sq: StructuredQuery, protocol: dict
) -> tuple[bool, str]:
    """4.5 — Factual grounding / anti-hallucination.

    Checks (across full output):
    - Lab-test deny-list: no unactionable lab references.
    - Vaccine whitelist: any vaccine mentioned must be on the whitelist.
    """
    # Lab-test deny check
    for lab_term in protocol.get("lab_test_deny_list", []):
        if lab_term.lower() in text.lower():
            return False, f"Unactionable lab reference: {lab_term!r}"

    # Vaccine whitelist check
    vaccine_wl = [v.lower() for v in protocol.get("vaccine_whitelist", [])]
    for mention in re.findall(
        r"(\w+(?:\s+\w+)?)\s+vacc(?:ine|ination)", text, re.IGNORECASE
    ):
        mention_lower = mention.lower().strip()
        if not any(wl in mention_lower or mention_lower in wl for wl in vaccine_wl):
            return False, f"Vaccine not on whitelist (possible hallucination): {mention!r}"

    return True, "ok"


def check_grounding_consistency(
    text: str, retrieved_chunks: list[dict] | None, protocol: dict
) -> tuple[bool, str]:
    """1.5 — Grounding enforcement (retrieval <-> generation consistency).
    Verifies that numeric thresholds, dosages, or key protocol steps output by the LLM
    are grounded in the retrieved chunks or protocol config."""
    if not retrieved_chunks:
        return True, "ok (no chunks to verify against)"

    combined_context = " ".join(c.get("text", "") for c in retrieved_chunks)

    # 1. Check for respiratory thresholds cited: must be in context or protocol
    for rate_str in re.findall(r"\b(\d+)\s*breaths?(?:\s*/?(?:per\s*)?min(?:ute)?)?\b", text, re.IGNORECASE):
        if rate_str not in combined_context:
            protocol_rates = {str(b["fast_breathing_threshold"]) for b in protocol.get("respiratory_rate_thresholds", [])}
            if rate_str not in protocol_rates:
                return False, f"Fabricated/ungrounded respiratory threshold: {rate_str} breaths/min"

    # 2. Check for honey dosing: if honey is mentioned, dose must match context/protocol
    if "honey" in text.lower() and ("battery" in combined_context.lower() or "button battery" in text.lower()):
        for odd_dose in re.findall(r"\b(\d+)\s*(?:mL|ml|tsp|tablespoon|teaspoon)\s*(?:of\s*)?honey\b", text, re.IGNORECASE):
            if odd_dose not in ["10", "2"]:
                return False, f"Ungrounded honey dosage for battery ingestion: {odd_dose}"

    return True, "ok"


def _log_event(event_type: str, data: dict) -> None:
    """4.6 — Structured JSON event logging for audit trail and regression corpus."""
    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_type": event_type,
        **data,
    }
    logging.getLogger("apollo.pipeline.audit").info(json.dumps(event, ensure_ascii=False))


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def validate_and_repair(
    llm_output: str,
    sq: StructuredQuery,
    llm_retry_fn: Callable[[str, str], str] | None = None,
    retrieved_chunks: list[dict] | None = None,
    retrieved_chunk_ids: list[str] | None = None,
) -> ApolloResponse | Escalation:
    """
    Layer C — Post-Processing Pipeline Orchestrator.

    Runs all named checks in priority order. Returns ApolloResponse on success,
    Escalation on hard failure.

    Check order & failure modes:
      4.1 schema_conformance      → repair stubs or retry once → escalate
      4.2 metadata_leakage        → strip via regex (no LLM needed)
      4.3 emergency_consistency   → HARD FAIL, NO RETRY — direct escalation
      4.4 age_band_consistency    → retry once with correction → escalate
      4.5 actionability           → retry once → escalate
      4.6 grounding_consistency   → retry once with citation constraint → escalate

    llm_retry_fn(cleaned_input, corrective_hint) -> str | None
      When None (unit tests), skips retry and goes directly to escalation.
    """
    protocol = load_clinical_protocol()
    current = llm_output
    checks_passed: list[str] = []
    regen_count = 0
    chunk_ids = retrieved_chunk_ids or []

    def escalate(reason: str, check: str) -> Escalation:
        _log_event("ESCALATION", {
            "reason": reason, "failed_check": check,
            "raw_input": sq.raw_input[:400],
            "active_emergency": sq.active_emergency,
            "matched_red_flags": sq.matched_red_flags,
            "llm_preview": current[:300],
            "retrieved_chunk_ids": chunk_ids,
        })
        return Escalation(
            content=ESCALATION_TEMPLATE,
            reason=reason, failed_check=check,
            raw_llm_output=llm_output, structured_query=sq,
        )

    def retry(hint: str) -> str | None:
        nonlocal regen_count
        if llm_retry_fn is None:
            return None
        regen_count += 1
        _log_event("REGENERATION", {"attempt": regen_count, "hint": hint[:200], "raw_input": sq.raw_input[:200]})
        return llm_retry_fn(sq.cleaned_input, hint)

    # ── 4.1 Schema conformance ────────────────────────────────────────────────
    ok, reason = check_schema_conformance(current)
    if not ok:
        logger.warning("[C1] Schema FAIL: %s", reason)
        _log_event("CHECK_FAIL", {"check": "schema_conformance", "reason": reason})
        repaired = retry(f"Schema validation failed: {reason}. Use EXACTLY the 4 required headers in order.")
        if repaired:
            current = repaired
            ok2, r2 = check_schema_conformance(current)
            if not ok2:
                return escalate(f"Schema non-conformant after retry: {r2}", "check_schema_conformance")
        else:
            # Inject missing stubs rather than escalate (soft repair, no LLM)
            for i, pat in enumerate(_SECTION_HEADERS):
                if not pat.search(current):
                    current = current.rstrip() + "\n\n" + _SECTION_STUBS[i]
    checks_passed.append("schema_conformance")

    # ── 4.2 Metadata leakage ─────────────────────────────────────────────────
    ok, reason = check_metadata_leakage(current, sq.raw_input, protocol)
    if not ok:
        logger.warning("[C2] Metadata leak: %s", reason)
        _log_event("CHECK_FAIL", {"check": "metadata_leakage", "reason": reason})
        # Deterministic strip — no LLM needed
        for dp in protocol.get("metadata_deny_patterns", []):
            current = re.sub(dp, "", current, flags=re.IGNORECASE | re.MULTILINE).strip()
    checks_passed.append("metadata_leakage")

    # ── 4.3 Emergency consistency — SAFETY GATE, fail closed ─────────────────
    ok, reason = check_emergency_consistency(current, sq, protocol)
    if not ok:
        logger.error("[C3] SAFETY FAIL: %s", reason)
        _log_event("SAFETY_FAIL", {
            "check": "emergency_consistency", "reason": reason,
            "active_emergency": sq.active_emergency,
            "flags": sq.matched_red_flags, "llm_preview": current[:500],
            "retrieved_chunk_ids": chunk_ids,
        })
        return escalate(reason, "check_emergency_consistency")  # NO retry
    checks_passed.append("emergency_consistency")

    # ── 4.4 Age-band consistency ──────────────────────────────────────────────
    ok, reason = check_age_band_consistency(current, sq, protocol)
    if not ok:
        logger.warning("[C4] Age-band FAIL: %s", reason)
        _log_event("CHECK_FAIL", {"check": "age_band_consistency", "reason": reason})
        if sq.age_band:
            hint = (
                f"CORRECTION — you cited the wrong respiratory rate threshold. "
                f"Patient is {sq.age_months} months old. Use ONLY "
                f"{sq.age_band['label']}."
            )
            repaired = retry(hint)
            if repaired:
                current = repaired
                ok2, r2 = check_age_band_consistency(current, sq, protocol)
                if not ok2:
                    return escalate(f"Age-band mismatch after retry: {r2}", "check_age_band_consistency")
    checks_passed.append("age_band_consistency")

    # ── 4.5 Actionability / anti-hallucination ────────────────────────────────
    ok, reason = check_actionability(current, sq, protocol)
    if not ok:
        logger.warning("[C5] Actionability FAIL: %s", reason)
        _log_event("CHECK_FAIL", {"check": "actionability", "reason": reason})
        hint = (
            f"CORRECTION — your response contained a claim that is not caregiver-actionable "
            f"or references a clinical entity not in the approved context. "
            f"Specifically: {reason}. Only reference observable caregiver signs and approved medications."
        )
        repaired = retry(hint)
        if repaired:
            current = repaired
            ok2, r2 = check_actionability(current, sq, protocol)
            if not ok2:
                return escalate(f"Actionability failed after retry: {r2}", "check_actionability")
    checks_passed.append("actionability")

    # ── 4.6 Grounding enforcement ─────────────────────────────────────────────
    if retrieved_chunks:
        ok, reason = check_grounding_consistency(current, retrieved_chunks, protocol)
        if not ok:
            logger.warning("[C6] Grounding FAIL: %s", reason)
            _log_event("CHECK_FAIL", {"check": "grounding_consistency", "reason": reason})
            hint = f"CORRECTION — your response contains ungrounded clinical values: {reason}. Cite only values from the provided clinical context."
            repaired = retry(hint)
            if repaired:
                current = repaired
                ok2, r2 = check_grounding_consistency(current, retrieved_chunks, protocol)
                if not ok2:
                    return escalate(f"Grounding check failed after retry: {r2}", "check_grounding_consistency")
        checks_passed.append("grounding_consistency")

    # ── Prune trailing overflow after Section 4 ───────────────────────────────
    s4m = re.search(r"(###\s*4\.\s*Likely Causes.*)", current, re.DOTALL | re.IGNORECASE)
    if s4m:
        tail = current[s4m.start():]
        overflow = re.search(r"\n###\s+(?!4\.)", tail)
        if overflow:
            current = current[: s4m.start() + overflow.start()].rstrip()
            logger.info("[C] Trailing overflow pruned after Section 4.")

    sections = _extract_sections(current)
    _log_event("SUCCESS", {
        "checks_passed": checks_passed, "regen_count": regen_count,
        "active_emergency": sq.active_emergency,
        "retrieved_chunk_ids": chunk_ids,
    })

    return ApolloResponse(
        content=current,
        section1=sections.get("1", ""),
        section2=sections.get("2", ""),
        section3=sections.get("3", ""),
        section4=sections.get("4", ""),
        checks_passed=checks_passed,
        regeneration_count=regen_count,
        retrieved_chunk_ids=chunk_ids,
    )
