"""
Fallback AI Scorer — Rule-Based

Activated when OpenAI is unavailable. Produces a deterministic score baseline
so the system degrades gracefully rather than crashing.
"""
from app.models.lead import Lead, LeadSource, ColdStage


def rule_based_score(lead: Lead) -> dict:
    """
    Compute a lead quality score using pure rule-based logic.
    Returns a dict with keys: score, recommendation, reason.
    All scores are in [0.0, 1.0].
    """
    score = 0.0
    reasons: list[str] = []

    # ── Source weight ─────────────────────────────
    source_weights = {
        LeadSource.REFERRAL: 0.35,
        LeadSource.PARTNER: 0.30,
        LeadSource.REGISTRATION: 0.28,
        LeadSource.WEB: 0.25,
        LeadSource.CALLBACK: 0.22,
        LeadSource.LEAD_MAGNET: 0.20,
        LeadSource.SCANNER: 0.20,
        LeadSource.SOCIAL: 0.15,
        LeadSource.MANUAL: 0.10,
        LeadSource.MESSAGE: 0.08,
    }
    src_w = source_weights.get(lead.source, 0.10)
    score += src_w
    reasons.append(f"source={lead.source.value}(+{src_w:.2f})")

    # ── Activity (message count) ──────────────────
    mc = lead.message_count if lead.message_count else 0
    if mc >= 10:
        score += 0.25
        reasons.append("high-activity")
    elif mc >= 5:
        score += 0.15
        reasons.append("medium-activity")
    elif mc >= 2:
        score += 0.08
        reasons.append("low-activity")

    # ── Contact completeness ──────────────────────
    if lead.email and lead.phone:
        score += 0.15
        reasons.append("full-contact")
    elif lead.email or lead.phone:
        score += 0.07
        reasons.append("partial-contact")

    # ── B2B qualification ─────────────────────────
    if getattr(lead, "company", None) and getattr(lead, "position", None):
        score += 0.10
        reasons.append("b2b-qualified")

    # ── Business domain ───────────────────────────
    if lead.business_domain:
        score += 0.15
        reasons.append("domain-set")

    # ── Stage weight ──────────────────────────────
    stage_weights = {
        ColdStage.QUALIFIED: 0.20,
        ColdStage.CONTACTED: 0.10,
        ColdStage.NEW: 0.0,
    }
    stg_w = stage_weights.get(lead.stage, 0.0)
    score += stg_w
    if stg_w > 0:
        reasons.append(f"stage={lead.stage.value}(+{stg_w:.2f})")

    # Cap at 1.0
    score = round(min(score, 1.0), 3)

    if score >= 0.6:
        recommendation = "transfer_to_sales"
    elif score >= 0.3:
        recommendation = "continue_nurturing"
    else:
        recommendation = "lost"

    reason = f"[RULE-BASED / AI OFFLINE] Signals: {', '.join(reasons) or 'none'}. Score: {score:.2f}."

    return {
        "score": score,
        "recommendation": recommendation,
        "reason": reason,
    }
