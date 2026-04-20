from __future__ import annotations

# Safety tests cover emergency category detection and escalation wording checks.

from firstaid_copilot.safety import assess_query, has_required_emergency_language


def test_safety_detects_high_risk_category():
    assessment = assess_query("What should I do if someone is not breathing and needs CPR?")
    assert assessment.risk_category == "cpr"
    assert assessment.call_emergency_now is True


def test_emergency_language_detection():
    assert has_required_emergency_language("Call emergency services now and start CPR.")
    assert not has_required_emergency_language("Start CPR and monitor them closely.")


def test_safety_detects_bee_sting_breathing_distress_as_emergency():
    assessment = assess_query(
        "I got stung by a bee, and now I have a rash and difficulty breathing."
    )
    assert assessment.risk_category == "allergies"
    assert assessment.call_emergency_now is True

