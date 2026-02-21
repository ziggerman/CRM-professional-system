"""
TransferService — orchestrates the lead → sale transfer.

This is the KEY business gate. It:
1. Validates AI score (hard threshold from config)
2. Validates business domain presence
3. Creates sale record
4. Updates lead stage to TRANSFERRED

The AI score is a NECESSARY condition but NOT sufficient alone.
Human must explicitly call the transfer endpoint.
"""
from datetime import datetime, UTC

from app.core.config import settings
from app.models.lead import Lead, ColdStage
from app.models.sale import Sale, SaleStage, SALE_STAGE_ORDER, TERMINAL_SALE_STAGES
from app.repositories.lead_repo import LeadRepository
from app.repositories.sale_repo import SaleRepository
from app.ai.ai_service import AIService
from app.schemas.lead import AIAnalysisResult


class TransferError(Exception):
    """Raised when transfer cannot be completed."""
    pass


class TransferService:
    def __init__(
        self,
        lead_repo: LeadRepository,
        sale_repo: SaleRepository,
        ai_service: AIService,
    ):
        self.lead_repo = lead_repo
        self.sale_repo = sale_repo
        self.ai_service = ai_service

    async def analyze_lead(self, lead: Lead) -> AIAnalysisResult:
        """
        Run AI analysis and persist result.
        Does NOT change lead stage or create sale.
        Manager sees the recommendation and decides what to do.
        """
        result = await self.ai_service.analyze_lead(lead)
        # Persist for auditability
        lead.ai_score = result.score
        lead.ai_recommendation = result.recommendation
        lead.ai_reason = result.reason
        lead.ai_analyzed_at = datetime.now(UTC)
        await self.lead_repo.save(lead)
        return result

    async def transfer_to_sales(self, lead: Lead, amount: int | None = None) -> tuple[Lead, Sale]:
        """
        Transfer lead to sales pipeline.

        Hard gates (enforced by code, NOT by AI):
        1. Lead must not already be transferred
        2. AI score must be >= MIN_TRANSFER_SCORE (default 0.6)
        3. Lead must have a business_domain set

        These rules are explicit and auditable.
        """
        # Gate 1: Stage check
        if lead.stage == ColdStage.TRANSFERRED:
            raise TransferError("Lead is already transferred to sales.")

        if lead.stage not in (ColdStage.QUALIFIED, ColdStage.CONTACTED, ColdStage.NEW):
            raise TransferError(f"Lead in stage '{lead.stage.value}' cannot be transferred.")

        # Gate 2: AI score check — we require a fresh analysis to exist
        if lead.ai_score is None:
            raise TransferError(
                "AI analysis required before transfer. Call /analyze first."
            )
        if lead.ai_score < settings.MIN_TRANSFER_SCORE:
            raise TransferError(
                f"AI score {lead.ai_score:.2f} is below minimum threshold "
                f"{settings.MIN_TRANSFER_SCORE}. Recommendation: {lead.ai_recommendation}."
            )

        # Gate 3: Business domain check
        if lead.business_domain is None:
            raise TransferError(
                "Lead must have a business_domain set before transfer to sales."
            )

        # All gates passed — execute transfer atomically
        lead.stage = ColdStage.TRANSFERRED
        await self.lead_repo.save(lead)

        sale = Sale(lead_id=lead.id, stage=SaleStage.NEW, amount=amount)
        sale = await self.sale_repo.create(sale)

        return lead, sale

    async def advance_sale_stage(self, sale: Sale, new_stage: SaleStage) -> Sale:
        """
        Move sale through its own pipeline with same sequential rules.
        """
        current = sale.stage

        # Check if terminal stage
        if current in TERMINAL_SALE_STAGES:
            raise TransferError(
                f"Sale is in terminal stage '{current.value}' and cannot be changed."
            )

        current_idx = SALE_STAGE_ORDER.index(current)
        new_idx = SALE_STAGE_ORDER.index(new_stage)

        # Can drop to LOST from any stage
        if new_stage == SaleStage.LOST:
            pass
        # Otherwise must be sequential
        elif new_idx != current_idx + 1:
            raise TransferError(
                f"Cannot transition sale from '{current.value}' to '{new_stage.value}'. "
                f"Expected next stage: '{SALE_STAGE_ORDER[current_idx + 1].value}'."
            )

        sale.stage = new_stage
        return await self.sale_repo.save(sale)
