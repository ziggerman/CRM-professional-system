"""
AI Prompts for Lead and Sale Analysis.

This module contains all system prompts, prompt builders, input validation,
and response parsing utilities for the AI service.
"""

import json
import logging
from typing import Optional, TypedDict

# Import enums directly so constants are always in sync with the domain model
from app.models.lead import LeadSource, ColdStage

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Type Definitions
# ──────────────────────────────────────────────

class LeadFeatures(TypedDict):
    source: str
    stage: str
    message_count: int
    days_since_created: int
    business_domain: Optional[str]


class SaleFeatures(TypedDict):
    stage: str
    days_in_stage: Optional[int]
    has_amount: Optional[bool]
    notes: Optional[str]


class LeadAnalysisResult(TypedDict):
    score: float
    recommendation: str
    reason: str


class SaleAnalysisResult(TypedDict):
    score: float
    recommendation: str
    reason: str


# ──────────────────────────────────────────────
# Constants — derived dynamically from model enums
# This ensures prompts.py is ALWAYS in sync with the domain model.
# ──────────────────────────────────────────────

# Built from LeadSource enum — any new source is automatically included
VALID_LEAD_SOURCES = frozenset(e.value for e in LeadSource)
# Built from ColdStage enum — replaces the old hardcoded NEGOTIATION/CLOSED
VALID_LEAD_STAGES = frozenset(e.value for e in ColdStage)
VALID_LEAD_RECOMMENDATIONS = frozenset({'transfer_to_sales', 'continue_nurturing', 'lost'})
REQUIRED_LEAD_FEATURES = frozenset({'source', 'stage', 'message_count', 'days_since_created'})

VALID_SALE_STAGES = frozenset({'NEW', 'KYC', 'AGREEMENT', 'PAID', 'LOST'})
VALID_SALE_RECOMMENDATIONS = frozenset({'priority', 'monitor', 'deprioritize'})
REQUIRED_SALE_FEATURES = frozenset({'stage'})


# ──────────────────────────────────────────────
# System Prompts
# ──────────────────────────────────────────────

LEAD_ANALYSIS_SYSTEM_PROMPT = """You are a senior CRM analyst specializing in lead qualification and scoring.
Your role is to analyze leads and provide a score (0.0–1.0) accompanied by a structured recommendation and reasoning.

## Scoring Guidelines
- **Score 0.0–0.3**: Low-quality lead — continue nurturing or mark as lost.
- **Score 0.3–0.6**: Medium-quality lead — requires additional nurturing before escalation.
- **Score 0.6–1.0**: High-quality lead — ready for transfer to the sales team.

## Key Scoring Factors
Evaluate all four factors holistically when determining the final score:

1. **Source** — Indicates lead origin and initial intent level.
   - `WEB` and `REFERRAL`: High-priority; strong intent signal.
   - `SOCIAL`: Moderate priority; requires identity and intent verification.
   - `MANUAL`: Low priority; typically cold with no prior engagement.

2. **Stage** — Reflects the lead's progression through the pipeline.
   - Stages must follow a logical sequence.
   - `QUALIFIED` and `CONTACTED` stages indicate higher purchase intent.

3. **Message Count** — Represents engagement depth over time.
   - Higher and rapidly increasing message counts are strong positive signals.

4. **Business Domain** — Critical for B2B relevance assessment.
   - The presence of a recognized domain (e.g., `TECHNOLOGY`, `RETAIL`) is a hard requirement for sales transfer.
   - Absence of a domain should significantly reduce the score.

## Output Format
Respond exclusively with valid JSON in the following structure:
```json
{
    "score": <float between 0.0 and 1.0>,
    "recommendation": "<one of: transfer_to_sales | continue_nurturing | lost>",
    "reason": "<2–3 sentence professional explanation referencing the four key factors above>"
}
```

## Behavioral Constraints
- Your role is **advisory only** — you do not make binding business decisions.
- All recommendations are suggestions to support human decision-making.
- Always provide a clear, factor-driven rationale for your score.
- Maintain consistent reasoning logic across leads with similar characteristics.
- Respond with JSON only — no additional text, markdown, or commentary outside the JSON block.
"""

SALE_QUALITY_SYSTEM_PROMPT = """You are a senior sales pipeline analyst specializing in deal quality assessment and close probability forecasting.
Your role is to evaluate active sales deals and predict their likelihood of successful closure.

## Scoring Guidelines
- **Score 0.0–0.3**: Low close probability — consider deprioritizing or reassigning.
- **Score 0.3–0.6**: Moderate close probability — monitor closely and apply targeted attention.
- **Score 0.6–1.0**: High close probability — prioritize for accelerated follow-up.

## Key Scoring Factors

1. **Stage** — Position in the sales pipeline indicates proximity to closure.
   - Later stages (`NEGOTIATION`, `CLOSING`) carry significantly higher weight.

2. **Days in Current Stage** — Time stagnation is a negative signal.
   - Deals stalled for extended periods should receive a lower score.

3. **Has Amount** — Presence of a defined deal value indicates seriousness and budget confirmation.
   - Missing amount is a negative signal regardless of stage.

4. **Notes** — Qualitative context can positively or negatively adjust the score.
   - Look for signals such as urgency, objections, competitor mentions, or stakeholder engagement.

## Output Format
Respond exclusively with valid JSON in the following structure:
```json
{
    "score": <float between 0.0 and 1.0>,
    "recommendation": "<one of: priority | monitor | deprioritize>",
    "reason": "<2–3 sentence professional explanation referencing the evaluated deal factors>"
}
```

## Behavioral Constraints
- Your role is **advisory only** — you do not make binding business decisions.
- Always provide a clear, factor-driven rationale for your score.
- Respond with JSON only — no additional text, markdown, or commentary outside the JSON block.
"""


# ──────────────────────────────────────────────
# Validation Helpers
# ──────────────────────────────────────────────

def _validate_lead_features(features: LeadFeatures) -> None:
    """
    Validate that all required lead fields are present and contain acceptable values.

    Args:
        features: Dictionary of lead attributes to validate.

    Raises:
        ValueError: If required fields are missing or contain invalid values.
    """
    missing = REQUIRED_LEAD_FEATURES - features.keys()
    if missing:
        raise ValueError(f"Missing required lead features: {missing}")

    if features['source'] not in VALID_LEAD_SOURCES:
        raise ValueError(
            f"Invalid lead source '{features['source']}'. "
            f"Must be one of: {VALID_LEAD_SOURCES}"
        )

    if features['stage'].upper() not in VALID_LEAD_STAGES:
        raise ValueError(
            f"Invalid lead stage '{features['stage']}'. "
            f"Must be one of: {VALID_LEAD_STAGES}"
        )

    if not isinstance(features['message_count'], int) or features['message_count'] < 0:
        raise ValueError("'message_count' must be a non-negative integer.")

    if not isinstance(features['days_since_created'], int) or features['days_since_created'] < 0:
        raise ValueError("'days_since_created' must be a non-negative integer.")


def _validate_sale_features(features: SaleFeatures) -> None:
    """
    Validate that all required sale fields are present and contain acceptable values.

    Args:
        features: Dictionary of sale deal attributes to validate.

    Raises:
        ValueError: If required fields are missing or contain invalid values.
    """
    missing = REQUIRED_SALE_FEATURES - features.keys()
    if missing:
        raise ValueError(f"Missing required sale features: {missing}")

    if features['stage'].upper() not in VALID_SALE_STAGES:
        raise ValueError(
            f"Invalid sale stage '{features['stage']}'. "
            f"Must be one of: {VALID_SALE_STAGES}"
        )

    days_in_stage = features.get('days_in_stage')
    if days_in_stage is not None and (not isinstance(days_in_stage, int) or days_in_stage < 0):
        raise ValueError("'days_in_stage' must be a non-negative integer.")


# ──────────────────────────────────────────────
# Prompt Builders
# ──────────────────────────────────────────────

def build_lead_analysis_prompt(features: LeadFeatures) -> str:
    """
    Validate and build the user prompt for lead scoring.

    Args:
        features: A typed dictionary containing lead attributes.

    Returns:
        A formatted prompt string ready for submission to the AI service.

    Raises:
        ValueError: If required fields are missing or values are invalid.
    """
    _validate_lead_features(features)

    return f"""Analyze the following lead and provide a score and recommendation based on the four key factors defined in your instructions.

## Lead Data
- **Source**: {features['source']}
- **Stage**: {features['stage']}
- **Message Count**: {features['message_count']}
- **Business Domain**: {features.get('business_domain') or 'Not specified'}
- **Days Since Created**: {features['days_since_created']}

Respond with your analysis in the required JSON format."""


def build_sale_analysis_prompt(features: SaleFeatures) -> str:
    """
    Validate and build the user prompt for deal quality analysis.

    Args:
        features: A typed dictionary containing sales deal attributes.

    Returns:
        A formatted prompt string ready for submission to the AI service.

    Raises:
        ValueError: If required fields are missing or values are invalid.
    """
    _validate_sale_features(features)

    return f"""Analyze the following sales deal and provide a close probability score and recommendation.

## Deal Data
- **Stage**: {features['stage']}
- **Days in Current Stage**: {features.get('days_in_stage', 0)}
- **Has Amount**: {features.get('has_amount', False)}
- **Notes**: {features.get('notes') or 'None'}

Respond with your analysis in the required JSON format."""


# ──────────────────────────────────────────────
# Response Parsers
# ──────────────────────────────────────────────

def parse_lead_analysis_response(raw: str) -> LeadAnalysisResult:
    """
    Parse and validate the AI response for lead analysis.

    Args:
        raw: Raw JSON string returned by the AI service.

    Returns:
        A validated LeadAnalysisResult dictionary.

    Raises:
        ValueError: If the response is malformed, missing fields, or contains invalid values.
    """
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        logger.error("Failed to parse lead analysis response as JSON: %s", raw)
        raise ValueError(f"AI response is not valid JSON: {e}") from e

    _validate_analysis_result(
        data,
        valid_recommendations=VALID_LEAD_RECOMMENDATIONS,
        context="lead",
    )

    return LeadAnalysisResult(
        score=data['score'],
        recommendation=data['recommendation'],
        reason=data['reason'],
    )


def parse_sale_analysis_response(raw: str) -> SaleAnalysisResult:
    """
    Parse and validate the AI response for sale quality analysis.

    Args:
        raw: Raw JSON string returned by the AI service.

    Returns:
        A validated SaleAnalysisResult dictionary.

    Raises:
        ValueError: If the response is malformed, missing fields, or contains invalid values.
    """
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        logger.error("Failed to parse sale analysis response as JSON: %s", raw)
        raise ValueError(f"AI response is not valid JSON: {e}") from e

    _validate_analysis_result(
        data,
        valid_recommendations=VALID_SALE_RECOMMENDATIONS,
        context="sale",
    )

    return SaleAnalysisResult(
        score=data['score'],
        recommendation=data['recommendation'],
        reason=data['reason'],
    )


def _validate_analysis_result(data: dict, valid_recommendations: frozenset, context: str) -> None:
    """
    Shared validation logic for AI analysis responses.

    Args:
        data: Parsed JSON dictionary from AI response.
        valid_recommendations: Set of acceptable recommendation values.
        context: Label used in error messages ('lead' or 'sale').

    Raises:
        ValueError: If any required field is missing or contains an invalid value.
    """
    required_fields = {'score', 'recommendation', 'reason'}
    missing = required_fields - data.keys()
    if missing:
        raise ValueError(f"AI {context} response missing required fields: {missing}")

    score = data['score']
    if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
        raise ValueError(
            f"AI {context} response 'score' must be a float between 0.0 and 1.0. Got: {score!r}"
        )

    recommendation = data['recommendation']
    if recommendation not in valid_recommendations:
        raise ValueError(
            f"AI {context} response 'recommendation' must be one of {valid_recommendations}. "
            f"Got: {recommendation!r}"
        )

    if not isinstance(data['reason'], str) or not data['reason'].strip():
        raise ValueError(f"AI {context} response 'reason' must be a non-empty string.")