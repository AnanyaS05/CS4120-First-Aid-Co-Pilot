from __future__ import annotations

from dataclasses import dataclass


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
    "allergies": ["allerg", "epipen", "anaphyla", "reaction", "hive"],
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


@dataclass(slots=True)
class SafetyAssessment:
    risk_category: str
    call_emergency_now: bool
    warnings: list[str]


def detect_risk_category(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return "other"


def assess_query(text: str) -> SafetyAssessment:
    category = detect_risk_category(text)
    warnings: list[str] = []
    call_emergency_now = category in EMERGENCY_CATEGORIES
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
    normalized = answer_text.lower()
    return any(marker in normalized for marker in EMERGENCY_LANGUAGE_MARKERS)

