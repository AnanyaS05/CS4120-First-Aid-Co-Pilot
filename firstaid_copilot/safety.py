from __future__ import annotations

# Deterministic safety checks for emergency categories and escalation language.

from dataclasses import dataclass


# Keyword matching is intentionally simple and auditable for this safety layer.
CATEGORY_KEYWORDS = {
    "choking": ["chok", "airway obstruct", "heimlich", "back blow"],
    "cpr": ["cpr", "cardiac arrest", "chest compression", "resuscitat", "aed", "defibrillator"],
    "severe_bleeding": ["bleed", "hemorrhag", "tourniquet", "pressure dressing"],
    "seizure": ["seizure", "convuls", "epilep", "fit "],
    "stroke": ["stroke", "face droop", "fast test", "brain attack"],
    "heart_attack": ["heart attack", "myocardial", "chest pain", "cardiac"],
    "burns": ["burn", "scald", "degree"],
    "fracture": ["fracture", "broken bone", "splint", "bone"],
    "spinal_injury": ["spinal", "spine", "neck injury", "vertebra"],
    "unconscious": ["unconscious", "unresponsive", "recovery position", "not breathing"],
    "transportation": ["human crutch", "drag", "move", "moving", "ambulance"],
    "allergies": [
        "allerg",
        "epipen",
        "anaphyla",
        "reaction",
        "hive",
        "rash",
        "bee sting",
        "wasp sting",
        "sting",
    ],
}

EMERGENCY_CATEGORIES = {
    "choking",
    "cpr",
    "severe_bleeding",
    "stroke",
    "heart_attack",
    "unconscious",
    "spinal_injury",
}

EMERGENCY_LANGUAGE_MARKERS = (
    "call emergency",
    "call 911",
    "call your local emergency number",
    "contact emergency services",
    "seek emergency medical help",
    "call for medical help",
)

BREATHING_DISTRESS_MARKERS = (
    "difficulty breathing",
    "trouble breathing",
    "shortness of breath",
    "cannot breathe",
    "can't breathe",
    "wheez",
    "swelling of the tongue",
    "swelling of tongue",
    "swelling of the throat",
    "swelling of throat",
)


@dataclass(slots=True)
class SafetyAssessment:
    risk_category: str
    call_emergency_now: bool
    warnings: list[str]


def detect_risk_category(text: str) -> str:
    """Return the first keyword-matched risk category for a query."""
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return "other"


def _has_emergency_signal(text_lower: str, category: str) -> bool:
    """Detect emergency clues that are not category membership alone."""
    if any(marker in text_lower for marker in BREATHING_DISTRESS_MARKERS):
        return True
    if (
        category == "allergies"
        and any(marker in text_lower for marker in ("bee sting", "wasp sting", "sting"))
        and any(marker in text_lower for marker in ("rash", "hive", "allerg", "anaphyla"))
    ):
        return True
    return False


def assess_query(text: str) -> SafetyAssessment:
    """Classify query risk and decide whether emergency language is required."""
    category = detect_risk_category(text)
    warnings: list[str] = []
    text_lower = text.lower()
    call_emergency_now = (
        category in EMERGENCY_CATEGORIES
        or _has_emergency_signal(text_lower, category)
    )
    if category == "other":
        warnings.append("The request did not clearly match a predefined high-risk category.")
    if call_emergency_now:
        warnings.append("Emergency escalation language should appear early in the answer.")
    return SafetyAssessment(
        risk_category=category,
        call_emergency_now=call_emergency_now,
        warnings=warnings,
    )


def has_required_emergency_language(answer_text: str) -> bool:
    """Check whether an answer includes required emergency escalation wording."""
    normalized = answer_text.lower()
    return any(marker in normalized for marker in EMERGENCY_LANGUAGE_MARKERS)

