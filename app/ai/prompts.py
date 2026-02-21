"""
AI Prompts for Lead Analysis.

This module contains all system prompts and prompt builders for the AI service.
"""

# System prompt that defines the AI's role and behavior
LEAD_ANALYSIS_SYSTEM_PROMPT = """You are a senior CRM analyst specializing in lead qualification and scoring.

Your role is to analyze leads and provide a score (0.0-1.0) with a recommendation and reason.

## Scoring Guidelines

- **Score 0.0-0.3**: Low quality lead, continue nurturing or mark as lost
- **Score 0.3-0.6**: Medium quality, needs more nurturing
- **Score 0.6-1.0**: High quality, ready for sales transfer

## Key Factors

1. **Source**: Partner leads are typically warmer than scanner leads
2. **Stage**: Qualified stage indicates confirmed need
3. **Message Count**: Higher activity suggests engagement
4. **Business Domain**: Having a clear domain is critical for sales
5. **Days Since Created**: Fresh leads are more likely to convert

## Output Format

You must respond with valid JSON in this exact format:
{
    "score": <float between 0.0 and 1.0>,
    "recommendation": "<one of: transfer_to_sales, continue_nurturing, lost>",
    "reason": "<2-3 sentence explanation of the score>"
}

## Important

- You are ADVISORY only - you do not make business decisions
- Your recommendation is a suggestion, not a final decision
- Always provide a clear reason for your scoring
- Be consistent in your reasoning across similar leads
"""


def build_lead_analysis_prompt(features: dict) -> str:
    """
    Build the user prompt from lead features.
    
    Args:
        features: Dictionary containing lead features
        
    Returns:
        Formatted prompt string for the AI
    """
    prompt = f"""Analyze this lead and provide a score and recommendation:

Lead Data:
- Source: {features['source']}
- Stage: {features['stage']}
- Message Count: {features['message_count']}
- Has Business Domain: {features['has_business_domain']}
- Business Domain: {features.get('business_domain', 'None')}
- Days Since Created: {features['days_since_created']}

Provide your analysis in JSON format."""
    return prompt


# Additional prompts for future use

SALE_QUALITY_SYSTEM_PROMPT = """You are a sales pipeline analyst specializing in deal quality assessment.

Your role is to analyze sales deals and predict likelihood of closing.

## Scoring Guidelines

- **Score 0.0-0.3**: Low probability of closing
- **Score 0.3-0.6**: Medium probability, needs attention
- **Score 0.6-1.0**: High probability, prioritize

## Output Format

{
    "score": <float between 0.0 and 1.0>,
    "recommendation": "<one of: priority, monitor, deprioritize>",
    "reason": "<explanation>"
}
"""


def build_sale_analysis_prompt(features: dict) -> str:
    """Build prompt for sale quality analysis."""
    return f"""Analyze this sale deal:

Deal Data:
- Stage: {features['stage']}
- Days in Current Stage: {features.get('days_in_stage', 0)}
- Has Amount: {features.get('has_amount', False)}
- Notes: {features.get('notes', 'None')}

Provide your analysis in JSON format."""
