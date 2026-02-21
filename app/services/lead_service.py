"""
LeadService — owns all business rules for lead lifecycle.

Rules enforced here (NOT in AI, NOT in API layer):
- Stage transitions must be sequential (no skipping)
- Terminal stages (transferred, lost) cannot be changed
- message_count can only increment
"""
from datetime import datetime, UTC

from app.models.lead import Lead, ColdStage, COLD_STAGE_ORDER, TERMINAL_COLD_STAGES
from app.repositories.lead_repo import LeadRepository
from app.schemas.lead import LeadCreate, AIAnalysisResult


class LeadStageError(Exception):
    """Raised when a stage transition is invalid."""
    pass


class LeadNotFoundError(Exception):
    """Raised when lead is not found."""
    pass


class LeadService:
    def __init__(self, lead_repo: LeadRepository):
        self.repo = lead_repo

    async def create_lead(self, data: LeadCreate) -> Lead:
        lead = Lead(
            source=data.source,
            business_domain=data.business_domain,
            telegram_id=data.telegram_id,
            stage=ColdStage.NEW,
            message_count=0,
        )
        return await self.repo.create(lead)

    async def get_lead(self, lead_id: int) -> Lead:
        lead = await self.repo.get_by_id(lead_id)
        if not lead:
            raise LeadNotFoundError(f"Lead {lead_id} not found")
        return lead

    async def get_leads(
        self, stage: ColdStage | None = None, offset: int = 0, limit: int = 50
    ) -> tuple[list[Lead], int]:
        return await self.repo.get_all(stage=stage, offset=offset, limit=limit)

    async def transition_stage(self, lead: Lead, new_stage: ColdStage) -> Lead:
        """
        Advance lead to next stage with strict validation.
        Only sequential moves allowed; terminal stages are locked.
        """
        current = lead.stage

        # Rule: terminal stages are immutable
        if current in TERMINAL_COLD_STAGES:
            raise LeadStageError(
                f"Lead is in terminal stage '{current.value}' and cannot be changed."
            )

        current_idx = COLD_STAGE_ORDER.index(current)
        new_idx = COLD_STAGE_ORDER.index(new_stage)

        # Rule: can only move forward by exactly one step (except → lost which is allowed from anywhere)
        if new_stage == ColdStage.LOST:
            pass  # Always allowed (drop lead)
        elif new_idx != current_idx + 1:
            raise LeadStageError(
                f"Cannot transition from '{current.value}' to '{new_stage.value}'. "
                f"Expected next stage: '{COLD_STAGE_ORDER[current_idx + 1].value}'."
            )

        lead.stage = new_stage
        return await self.repo.save(lead)

    async def increment_messages(self, lead: Lead, count: int = 1) -> Lead:
        lead.message_count += count
        return await self.repo.save(lead)

    async def save_ai_analysis(self, lead: Lead, result: AIAnalysisResult) -> Lead:
        """Persist AI result to lead. Does NOT trigger any state change."""
        lead.ai_score = result.score
        lead.ai_recommendation = result.recommendation
        lead.ai_reason = result.reason
        lead.ai_analyzed_at = datetime.now(UTC)
        return await self.repo.save(lead)
