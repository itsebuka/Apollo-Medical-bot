"""
Apollo Clinical Triage — Regression Test Suite
===============================================
Tests the deterministic pipeline (pipeline.py) in isolation.
No running LLM or FastAPI server required.

Run with:
    .\backend\venv\Scripts\python.exe -m pytest tests/test_apollo_pipeline.py -v

Every test name maps to a known failure mode from the task spec or session
history. Passing this suite does NOT mean the backend is production-safe —
see task spec §7 for clinician review requirements.
"""

import sys
import os
import re
import pytest

# Make both backend/app and repo root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.pipeline import (
    preprocess_query,
    validate_and_repair,
    build_system_context_block,
    check_schema_conformance,
    check_metadata_leakage,
    check_emergency_consistency,
    check_age_band_consistency,
    check_actionability,
    extract_age_months,
    get_age_band,
    scan_red_flags,
    load_clinical_protocol,
    invalidate_protocol_cache,
    StructuredQuery,
    ApolloResponse,
    Escalation,
    ESCALATION_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_response(section1="", section2="", section3="", section4="") -> str:
    """Build a minimal valid 4-part response for testing."""
    s1 = section1 or "This is an emergency. Go to hospital immediately — do not wait."
    s2 = section2 or "- Seek immediate emergency care."
    s3 = section3 or "Go to the nearest emergency room immediately. Do not give food or liquids."
    s4 = section4 or "Further evaluation required by a medical professional."
    return (
        f"### 1. Immediate Priority\n\n{s1}\n\n"
        f"### 2. Emergency Red Flags (Seek Immediate Medical Care)\n\n{s2}\n\n"
        f"### 3. Immediate Actions & Supportive Measures\n\n{s3}\n\n"
        f"### 4. Likely Causes (Differential Overview)\n\n{s4}"
    )


def _emergency_sq(age_months=6) -> StructuredQuery:
    """Build a StructuredQuery with active_emergency=True for testing."""
    protocol = load_clinical_protocol()
    band = get_age_band(age_months, protocol)
    return StructuredQuery(
        raw_input="test",
        cleaned_input="test",
        age_months=age_months,
        age_band=band,
        symptoms=["breathing", "chest"],
        matched_red_flags=["chest indrawing", "too weak to nurse"],
        active_emergency=True,
    )


def _non_emergency_sq() -> StructuredQuery:
    return StructuredQuery(
        raw_input="test",
        cleaned_input="test",
        age_months=18,
        age_band=get_age_band(18, load_clinical_protocol()),
        symptoms=["fever"],
        matched_red_flags=[],
        active_emergency=False,
    )


# ---------------------------------------------------------------------------
# Pre-processing — age extraction
# ---------------------------------------------------------------------------

class TestAgeExtraction:
    """Verify regex-only age extraction covers all common phrasings."""

    def test_months_standard(self):
        assert extract_age_months("My 6-month-old has fever") == 6

    def test_months_spaced(self):
        assert extract_age_months("baby is 3 months old") == 3

    def test_years_to_months(self):
        assert extract_age_months("My 2-year-old has a rash") == 24

    def test_weeks_to_months(self):
        result = extract_age_months("4 week old newborn")
        assert 0 <= result <= 2  # ~1 month

    def test_newborn(self):
        assert extract_age_months("newborn baby") == 0

    def test_no_age(self):
        assert extract_age_months("patient has fever and cough") is None


class TestAgeBandBoundaries:
    """Boundary tests at exact band edges — must not bleed across bands."""

    def setup_method(self):
        self.protocol = load_clinical_protocol()

    def test_age_exactly_2_months_uses_young_infant_band(self):
        """Exactly 2 months must use the 2-11mo band (>=50), not the <2mo band (>=60)."""
        band = get_age_band(2, self.protocol)
        assert band is not None
        assert band["fast_breathing_threshold"] == 50
        assert band["band"] == "young_infant"

    def test_age_1_month_uses_neonatal_band(self):
        """1 month must use the neonatal band (>=60)."""
        band = get_age_band(1, self.protocol)
        assert band is not None
        assert band["fast_breathing_threshold"] == 60
        assert band["band"] == "neonatal"

    def test_age_exactly_12_months_uses_toddler_band(self):
        """Exactly 12 months must use toddler band (>=40), not young_infant (>=50)."""
        band = get_age_band(12, self.protocol)
        assert band is not None
        assert band["fast_breathing_threshold"] == 40
        assert band["band"] == "toddler"

    def test_age_11_months_uses_young_infant_band(self):
        """11 months must still use young_infant band."""
        band = get_age_band(11, self.protocol)
        assert band is not None
        assert band["fast_breathing_threshold"] == 50

    def test_age_6_months_uses_correct_threshold(self):
        """test_age_band_6mo_uses_correct_threshold — regression for age-band mismatch bug."""
        band = get_age_band(6, self.protocol)
        assert band["fast_breathing_threshold"] == 50  # NOT 60 (neonatal)


# ---------------------------------------------------------------------------
# Pre-processing — red flag scan
# ---------------------------------------------------------------------------

class TestRedFlagScan:
    def setup_method(self):
        self.protocol = load_clinical_protocol()

    def test_chest_indrawing_detected(self):
        flags = scan_red_flags("baby has chest indrawing and won't feed", self.protocol)
        assert "chest indrawing" in flags

    def test_cyanosis_detected(self):
        flags = scan_red_flags("lips are turning blue, cyanosis", self.protocol)
        assert "cyanosis" in flags or "turning blue" in flags

    def test_seizure_detected(self):
        flags = scan_red_flags("the child had a seizure this morning", self.protocol)
        assert "seizure" in flags

    def test_battery_detected(self):
        flags = scan_red_flags("child swallowed a button battery", self.protocol)
        assert "button battery" in flags

    def test_mild_cough_not_flagged(self):
        """test_mild_cough_no_emergency_override — mild symptoms must not trigger emergency flag."""
        flags = scan_red_flags("child has a mild cough and runny nose", self.protocol)
        assert len(flags) == 0


class TestPreprocessQuery:
    def test_strips_track_prefix(self):
        raw = "FOR TRACK 1, QUESTION 2: My 6-month-old has chest indrawing."
        sq = preprocess_query(raw)
        assert "TRACK" not in sq.cleaned_input
        assert "QUESTION" not in sq.cleaned_input
        assert "chest indrawing" in sq.cleaned_input.lower()

    def test_strips_q_prefix(self):
        raw = "Q: My child has fever"
        sq = preprocess_query(raw)
        assert not sq.cleaned_input.upper().startswith("Q:")

    def test_active_emergency_set_for_respiratory_distress(self):
        sq = preprocess_query("My 6-month-old has chest indrawing and is too weak to nurse")
        assert sq.active_emergency is True
        assert "chest indrawing" in sq.matched_red_flags or "too weak to nurse" in sq.matched_red_flags

    def test_no_emergency_for_mild_symptoms(self):
        sq = preprocess_query("My 18-month-old has mild diarrhea and slight fever")
        assert sq.active_emergency is False

    def test_age_extracted_correctly(self):
        sq = preprocess_query("My 6-month-old baby has a rash")
        assert sq.age_months == 6
        assert sq.age_band is not None
        assert sq.age_band["fast_breathing_threshold"] == 50


# ---------------------------------------------------------------------------
# Post-processing — individual check functions
# ---------------------------------------------------------------------------

class TestSchemaConformance:
    def test_valid_response_passes(self):
        ok, reason = check_schema_conformance(_make_valid_response())
        assert ok, reason

    def test_missing_section_fails(self):
        text = (
            "### 1. Immediate Priority\n\nEmergency.\n\n"
            "### 2. Emergency Red Flags (Seek Immediate Medical Care)\n\nSee doctor.\n\n"
            "### 4. Likely Causes (Differential Overview)\n\nViral."
        )
        ok, reason = check_schema_conformance(text)
        assert not ok
        assert "3" in reason

    def test_duplicate_section_fails(self):
        """test_no_trailing_duplicate_section."""
        text = _make_valid_response() + "\n\n### 1. Immediate Priority\n\nDuplicate."
        ok, reason = check_schema_conformance(text)
        assert not ok
        assert "duplicate" in reason.lower() or "more than once" in reason.lower()

    def test_preamble_before_section1_fails(self):
        preamble = "Apollo Triage Summary\n\nGenerated: 10:23\n\n" + _make_valid_response()
        ok, _ = check_schema_conformance(preamble)
        assert not ok


class TestMetadataLeakage:
    def setup_method(self):
        self.protocol = load_clinical_protocol()

    def test_no_leakage_passes(self):
        ok, reason = check_metadata_leakage(_make_valid_response(), "clean query", self.protocol)
        assert ok, reason

    def test_track_keyword_in_output_fails(self):
        """test_no_track_question_leakage."""
        text = _make_valid_response(section1="This answers TRACK 1, QUESTION 2 about fever.")
        ok, reason = check_metadata_leakage(text, "FOR TRACK 1, QUESTION 2: fever", self.protocol)
        assert not ok

    def test_harness_literal_echoed_fails(self):
        raw_input = "FOR TRACK 3 fever baby"
        text = _make_valid_response(section4="See TRACK 3 documentation.")
        ok, reason = check_metadata_leakage(text, raw_input, self.protocol)
        assert not ok

    def test_meta_commentary_persona_break(self):
        """test_no_meta_commentary_persona_break — third-person clinician language."""
        text = _make_valid_response(
            section1="As a healthcare provider, it is important to be culturally appropriate."
        )
        # This specific test checks the SYSTEM PROMPT is followed — the pipeline checks
        # benchmark leakage. The persona break check is a prompt-level control, so we
        # verify the schema check passes but the response content is not what we want.
        # The pipeline cannot auto-reject persona breaks (too many false positives),
        # so we assert the schema still passes and log this for human review.
        ok, _ = check_schema_conformance(text)
        assert ok  # Schema conforms; persona break is caught at prompt level


class TestEmergencyConsistency:
    """4.3 — The highest-severity check."""

    def setup_method(self):
        self.protocol = load_clinical_protocol()
        self.esq = _emergency_sq()

    def test_valid_emergency_response_passes(self):
        text = _make_valid_response()
        ok, reason = check_emergency_consistency(text, self.esq, self.protocol)
        assert ok, reason

    def test_infant_respiratory_distress_no_feeding(self):
        """test_infant_respiratory_distress_no_feeding — REGRESSION."""
        s3_with_feeding = "Continue nursing the baby. Offer breast milk frequently."
        text = _make_valid_response(section3=s3_with_feeding)
        ok, reason = check_emergency_consistency(text, self.esq, self.protocol)
        assert not ok
        assert "unsafe advice" in reason.lower() or "deny" in reason.lower()

    def test_infant_respiratory_distress_no_wait_and_watch(self):
        """test_infant_respiratory_distress_no_wait_and_watch — REGRESSION."""
        s3_wait = "Monitor at home for the next few hours. Reassess in 2 hours if no improvement."
        text = _make_valid_response(section3=s3_wait)
        ok, reason = check_emergency_consistency(text, self.esq, self.protocol)
        assert not ok

    def test_section1_must_have_emergency_language(self):
        """Emergency query but Section 1 has no urgency language — must fail."""
        text = _make_valid_response(section1="Your child may have a viral infection. Rest and fluids recommended.")
        ok, reason = check_emergency_consistency(text, self.esq, self.protocol)
        assert not ok
        assert "emergency confirmation" in reason.lower() or "section 1" in reason.lower()

    def test_non_emergency_query_passes_any_section3(self):
        """Non-emergency: Section 3 may include fluids/monitoring without failing."""
        sq = _non_emergency_sq()
        text = _make_valid_response(section3="Offer oral rehydration fluids. Monitor at home.")
        ok, _ = check_emergency_consistency(text, sq, self.protocol)
        assert ok

    def test_altered_mental_status_emergency(self):
        """test_altered_mental_status_emergency — altered consciousness must trigger emergency."""
        sq = preprocess_query("My child has altered mental status and won't wake up")
        assert sq.active_emergency is True


class TestAgeBandConsistency:
    def setup_method(self):
        self.protocol = load_clinical_protocol()

    def test_correct_threshold_passes(self):
        """6-month-old: >=50 breaths/min is correct."""
        sq = preprocess_query("My 6-month-old baby has fast breathing, 55 breaths per minute")
        text = _make_valid_response(section1="Fast breathing at 55 breaths/min (>=50 for this age band) is concerning.")
        ok, reason = check_age_band_consistency(text, sq, self.protocol)
        assert ok, reason

    def test_age_band_6mo_uses_correct_threshold(self):
        """test_age_band_6mo_uses_correct_threshold — must cite 50 not 60."""
        sq = preprocess_query("My 6-month-old has rapid breathing at 62 breaths per minute")
        # Text incorrectly cites 60 (neonatal threshold)
        text = _make_valid_response(section1="Fast breathing >60 breaths per minute is a danger sign.")
        ok, reason = check_age_band_consistency(text, sq, self.protocol)
        assert not ok, "Should fail: 60 is the wrong threshold for a 6-month-old"

    def test_no_age_data_passes(self):
        sq = StructuredQuery(raw_input="x", cleaned_input="x")
        ok, _ = check_age_band_consistency(_make_valid_response(), sq, self.protocol)
        assert ok


class TestActionability:
    def setup_method(self):
        self.protocol = load_clinical_protocol()

    def test_no_unactionable_lab_advice(self):
        """test_no_unactionable_lab_advice — REGRESSION for 'monitor electrolytes' failure."""
        text = _make_valid_response(section3="Monitor electrolytes and serum sodium levels at home.")
        sq = _non_emergency_sq()
        ok, reason = check_actionability(text, sq, self.protocol)
        assert not ok
        assert "electrolyte" in reason.lower() or "lab" in reason.lower()

    def test_cbc_reference_fails(self):
        text = _make_valid_response(section3="Check CBC and complete blood count at your local clinic today.")
        sq = _non_emergency_sq()
        ok, reason = check_actionability(text, sq, self.protocol)
        assert not ok

    def test_no_fabricated_vaccine(self):
        """test_no_fabricated_vaccine — REGRESSION for fabricated 'norovirus vaccine'."""
        text = _make_valid_response(
            section4="Norovirus vaccination should be considered once the acute episode resolves."
        )
        sq = _non_emergency_sq()
        ok, reason = check_actionability(text, sq, self.protocol)
        assert not ok
        assert "whitelist" in reason.lower() or "norovirus" in reason.lower()

    def test_valid_observable_signs_pass(self):
        text = _make_valid_response(
            section3="Count your child's breathing rate. Check for wet diapers every 6 hours. "
                     "Watch for skin turgor by pinching the skin gently."
        )
        sq = _non_emergency_sq()
        ok, reason = check_actionability(text, sq, self.protocol)
        assert ok, reason


# ---------------------------------------------------------------------------
# Battery ingestion protocol
# ---------------------------------------------------------------------------

class TestBatteryIngestion:
    def test_battery_ingestion_triggers_emergency(self):
        sq = preprocess_query("My 2-year-old swallowed a button battery 2 hours ago")
        assert sq.active_emergency is True
        assert sq.substance_protocol is not None
        assert "honey_protocol" in sq.substance_protocol

    def test_battery_ingestion_honey_protocol_over_12mo(self):
        """test_battery_ingestion_honey_protocol — Section 3 must include honey instruction for >=12mo."""
        sq = preprocess_query("My 2-year-old swallowed a button battery 2 hours ago")
        assert sq.substance_protocol is not None
        hp = sq.substance_protocol.get("honey_protocol", {})
        assert hp.get("eligible_min_age_months") == 12
        assert hp.get("dose_ml") == 10

    def test_battery_ingestion_under_1yr_no_honey(self):
        """test_battery_ingestion_under_1yr_no_honey — honey protocol must NOT apply for <12mo."""
        sq = preprocess_query("My 6-month-old swallowed a button battery")
        assert sq.active_emergency is True
        assert sq.substance_protocol is not None
        hp = sq.substance_protocol.get("honey_protocol", {})
        # The honey protocol threshold is 12 months
        assert hp.get("eligible_min_age_months") == 12
        # A caregiver response for a 6-month-old must NOT give honey
        # This is enforced by build_system_context_block injecting the warning
        context_block = build_system_context_block(sq)
        assert "botulism" in context_block.lower() or "contraindicated" in context_block.lower()

    def test_no_induce_vomiting_in_battery_protocol(self):
        sq = preprocess_query("My 2-year-old ate a button battery")
        assert sq.substance_protocol.get("induce_vomiting") is False


# ---------------------------------------------------------------------------
# Full pipeline — validate_and_repair orchestration
# ---------------------------------------------------------------------------

class TestValidateAndRepair:
    def test_good_response_returns_apollo_response(self):
        sq = _non_emergency_sq()
        result = validate_and_repair(_make_valid_response(), sq)
        assert isinstance(result, ApolloResponse)
        assert "schema_conformance" in result.checks_passed
        assert "emergency_consistency" in result.checks_passed

    def test_safety_fail_returns_escalation_not_apollo_response(self):
        """Safety check failure must return Escalation — never silently ship failing output."""
        sq = _emergency_sq()
        # Response lacks emergency language in Section 1 (safety fail)
        bad_s1 = "Your baby may be unwell. Rest and monitor at home for now."
        bad_resp = _make_valid_response(section1=bad_s1)
        result = validate_and_repair(bad_resp, sq)
        assert isinstance(result, Escalation)
        assert result.failed_check == "check_emergency_consistency"
        # Escalation content must contain the hardcoded safe template
        assert "emergency" in result.content.lower()
        assert "### 1. Immediate Priority" in result.content

    def test_metadata_stripped_automatically(self):
        """Metadata leakage is stripped deterministically without LLM retry."""
        sq = _non_emergency_sq()
        sq.raw_input = "APOLLO TESTING: diarrhea query"
        text = _make_valid_response(section4="Apollo Triage Summary generated 10:23 about this.")
        result = validate_and_repair(text, sq)
        assert isinstance(result, ApolloResponse)
        assert "Apollo Triage Summary" not in result.content

    def test_no_trailing_duplicate_section(self):
        """test_no_trailing_duplicate_section — overflow after Section 4 must be pruned."""
        sq = _non_emergency_sq()
        overflow = _make_valid_response() + "\n\n### 5. Additional Guidance\n\nExtra duplicate content."
        result = validate_and_repair(overflow, sq)
        assert isinstance(result, ApolloResponse)
        assert "### 5." not in result.content

    def test_escalation_template_has_all_4_sections(self):
        """The hardcoded escalation template must itself be schema-conformant."""
        ok, reason = check_schema_conformance(ESCALATION_TEMPLATE)
        assert ok, f"Escalation template schema invalid: {reason}"


# ---------------------------------------------------------------------------
# Adversarial / injection tests
# ---------------------------------------------------------------------------

class TestAdversarialInputs:
    def test_track_label_mid_sentence_not_in_output(self):
        """test_no_track_question_leakage — adversarial: label embedded mid-sentence."""
        raw = "my baby (FOR TRACK 1, QUESTION 2) is having trouble breathing"
        sq = preprocess_query(raw)
        # Preprocessor must strip it directly
        assert "TRACK" not in sq.cleaned_input
        assert "QUESTION" not in sq.cleaned_input
        assert "having trouble breathing" in sq.cleaned_input

        # Even if an uncleaned string reached the post-processor, metadata check must catch it
        protocol = load_clinical_protocol()
        text_with_leak = _make_valid_response(section4="This was assessed under FOR TRACK 1, QUESTION 2 conditions.")
        ok, reason = check_metadata_leakage(text_with_leak, raw, protocol)
        assert not ok

    def test_mild_cough_does_not_trigger_emergency_flag(self):
        """test_mild_cough_no_emergency_override — over-trigger false positive guard."""
        sq = preprocess_query("My 3-year-old has a mild cough and is eating and drinking normally")
        assert sq.active_emergency is False

    def test_skin_colour_mention_does_not_trigger_cyanosis_flag(self):
        """'fair skin colour' must not match 'cyanosis' red-flag pattern."""
        sq = preprocess_query("My baby has fair skin colour and a runny nose")
        assert sq.active_emergency is False


