"""
LeadService — owns all business rules for lead lifecycle.
Step 5 — Business Rules & Pipeline Validation

Rules enforced here (NOT in AI, NOT in API layer):
- Stage transitions must be sequential (no skipping)
- Terminal stages (transferred, lost) cannot be changed
- message_count can only increment
- Mandatory fields validation per stage
- Lost reason taxonomy enforcement
"""
from datetime import datetime, UTC
from typing import Dict, Set, Optional

from app.models.lead import (
    Lead,
    ColdStage,
    LostReason,
    COLD_STAGE_ORDER,
    TERMINAL_COLD_STAGES,
    REVERSIBLE_STAGE_TRANSITIONS,
    calculate_quality_tier,
)
from app.models.history import LeadHistory
from app.repositories.lead_repo import LeadRepository
from app.repositories.history_repo import HistoryRepository
from app.repositories.user_repo import UserRepository
from app.schemas.lead import LeadCreate, AIAnalysisResult, LeadAttachmentResponse
from app.models.attachment import LeadAttachment
from app.api.v1.ws import manager as ws_manager


class LeadStageError(Exception):
    """Raised when a stage transition is invalid."""
    pass


class LeadNotFoundError(Exception):
    """Raised when lead is not found."""
    pass


class DuplicateLeadError(Exception):
    """Raised when a lead with the same email or phone already exists."""
    def __init__(self, existing_id: int, field: str):
        self.existing_id = existing_id
        self.field = field
        super().__init__(f"Duplicate lead detected by {field}. Existing lead ID: {existing_id}")


class MandatoryFieldsError(Exception):
    """Raised when mandatory fields are missing for stage transition."""
    def __init__(self, stage: ColdStage, missing_fields: list):
        self.stage = stage
        self.missing_fields = missing_fields
        super().__init__(f"Cannot transition to {stage.value}. Missing required fields: {', '.join(missing_fields)}")


# ─────────────────────────────────────────────────────────────
# Stage 5: Mandatory Fields per Stage
# These fields are REQUIRED before a lead can advance to the next stage
# ─────────────────────────────────────────────────────────────

# Fields required to move from NEW → CONTACTED (must have at least one contact)
STAGE_REQUIREMENTS: Dict[ColdStage, Dict[str, Set[str]]] = {
    ColdStage.NEW: {},  # No requirements for creation
    ColdStage.CONTACTED: {
        "any_of": {"phone", "email", "telegram_id"}  # At least one contact method
    },
    ColdStage.QUALIFIED: {
        "required": {"full_name", "business_domain"},  # Full name and domain required
        "any_of": {"phone", "email", "telegram_id"}   # Plus contact
    },
    ColdStage.TRANSFERRED: {
        "required": {"full_name", "business_domain", "ai_score"},  # Must be qualified + AI analyzed
        "any_of": {"phone", "email", "telegram_id"}
    },
    ColdStage.LOST: {},  # Can lose at any stage
}


def validate_stage_transition(lead: Lead, new_stage: ColdStage) -> list[str]:
    """
    Validate that lead has all required fields for the target stage.
    Returns list of missing fields (empty if valid).
    """
    if new_stage not in STAGE_REQUIREMENTS:
        return []
    
    requirements = STAGE_REQUIREMENTS[new_stage]
    missing = []
    
    # Check "required" fields (all must be present)
    required_fields = requirements.get("required", set())
    for field in required_fields:
        value = getattr(lead, field, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    
    # Check "any_of" fields (at least one must be present)
    any_of_fields = requirements.get("any_of", set())
    if any_of_fields:
        has_any = False
        for field in any_of_fields:
            value = getattr(lead, field, None)
            if value is not None and (not isinstance(value, str) or value.strip()):
                has_any = True
                break
        if not has_any:
            missing.extend([f for f in any_of_fields if not getattr(lead, f, None)])
    
    return missing


class LeadService:
    def __init__(self, lead_repo: LeadRepository, history_repo: "HistoryRepository"):
        self.repo = lead_repo
        self.history_repo = history_repo

    async def create_lead(self, data: LeadCreate) -> Lead:
        assigned_to_id = None

        async with self.repo.db.begin_nested():
            # Auto-assignment (round-robin by active manager load)
            user_repo = UserRepository(self.repo.db)
            preferred_domain = data.business_domain.value if data.business_domain else None
            manager = await user_repo.get_round_robin_manager(preferred_domain)
            if manager:
                assigned_to_id = manager.id
                manager.current_leads += 1
                manager.last_lead_assigned_at = datetime.now(UTC)
                await user_repo.save(manager)

            lead = Lead(
                full_name=data.full_name,
                phone=data.phone,
                email=data.email,
                source=data.source,
                business_domain=data.business_domain,
                telegram_id=data.telegram_id,
                stage=ColdStage.NEW,
                message_count=0,
                assigned_to_id=assigned_to_id,
            )
            lead = await self.repo.create(lead)
        
        # Broadcast update (Step 8.2)
        await ws_manager.broadcast({
            "type": "LEAD_CREATED",
            "id": lead.id,
            "stage": lead.stage.value,
            "assigned_to_id": lead.assigned_to_id,
        })
        
        return lead

    async def get_lead(self, lead_id: int, include_deleted: bool = False) -> Lead:
        lead = await self.repo.get_by_id(lead_id)
        if not lead:
            raise LeadNotFoundError(f"Lead {lead_id} not found")
        # By default, don't return deleted leads
        if not include_deleted and lead.is_deleted:
            raise LeadNotFoundError(f"Lead {lead_id} not found")
        return lead

    async def get_leads(
        self,
        stage: ColdStage | None = None,
        source: str | None = None,
        business_domain: str | None = None,
        assigned_to_id: int | None = None,
        telegram_id: str | None = None,
        query: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Lead], int]:
        return await self.repo.get_all(
            stage=stage,
            source=source,
            business_domain=business_domain,
            assigned_to_id=assigned_to_id,
            telegram_id=telegram_id,
            query=query,
            created_after=created_after,
            created_before=created_before,
            offset=offset,
            limit=limit,
        )

    async def transition_stage(
        self,
        lead: Lead,
        new_stage: ColdStage,
        changed_by: str = "System",
        lost_reason: LostReason | None = None,
    ) -> Lead:
        """
        Advance lead to next stage with strict validation.
        Step 5 — Validates mandatory fields per stage.
        
        Only sequential moves allowed; terminal stages are locked.
        Automatically logs the transition to HistoryRepository.
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

        # Step 5: Validate mandatory fields for the target stage
        if new_stage != ColdStage.LOST:
            missing_fields = validate_stage_transition(lead, new_stage)
            if missing_fields:
                raise MandatoryFieldsError(new_stage, missing_fields)

        # Business rule: LOST stage must always have structured reason
        if new_stage == ColdStage.LOST and lost_reason is None:
            raise LeadStageError("lost_reason is required when moving lead to LOST stage.")

        if new_stage != ColdStage.LOST and lost_reason is not None:
            raise LeadStageError("lost_reason can only be provided when moving to LOST stage.")

        # Log history before saving
        history_reason = f"Transitioned to {new_stage.value}"
        if new_stage == ColdStage.LOST and lost_reason is not None:
            history_reason = f"Transitioned to LOST ({lost_reason.value})"

        history = LeadHistory(
            lead_id=lead.id,
            old_stage=current.value,
            new_stage=new_stage.value,
            changed_by=changed_by,
            reason=history_reason,
        )
        
        # Save both inside the same transaction
        self.repo.db.add(history)
        lead.stage = new_stage
        if new_stage == ColdStage.LOST:
            lead.lost_reason = lost_reason
        updated_lead = await self.repo.save(lead)
        await ws_manager.broadcast({"event": "lead_updated", "lead_id": lead.id, "stage": new_stage.value})
        return updated_lead

    async def increment_messages(self, lead: Lead, count: int = 1) -> Lead:
        lead.message_count += count
        return await self.repo.save(lead)

    async def update_lead(self, lead: Lead, data: "LeadUpdate") -> Lead:
        """Update lead details."""
        if data.full_name is not None:
            lead.full_name = data.full_name
        if data.phone is not None:
            lead.phone = data.phone
        if data.email is not None:
            lead.email = data.email
        if data.source is not None:
            lead.source = data.source
        if data.business_domain is not None:
            lead.business_domain = data.business_domain
            
        return await self.repo.save(lead)

    async def save_ai_analysis(self, lead: Lead, result: AIAnalysisResult, analyzed_by: str = "openai") -> Lead:
        """Persist AI result to lead. Does NOT trigger any state change."""
        from app.models.score_history import LeadScoreHistory
        
        # Save to score history for audit trail
        score_history = LeadScoreHistory(
            lead_id=lead.id,
            score=result.score,
            recommendation=result.recommendation,
            reason=result.reason,
            analyzed_by=analyzed_by,
        )
        self.repo.db.add(score_history)
        
        # Update lead with new score and calculate quality tier
        lead.ai_score = result.score
        lead.ai_recommendation = result.recommendation
        lead.ai_reason = result.reason
        lead.ai_analyzed_at = datetime.now(UTC)
        lead.quality_tier = calculate_quality_tier(result.score)
        
        return await self.repo.save(lead)

    async def rollback_stage(self, lead: Lead, reason: str, changed_by: str = "System") -> Lead:
        """
        Rollback lead to previous stage.
        
        Only allowed for reversible transitions defined in REVERSIBLE_STAGE_TRANSITIONS.
        Requires a reason (min 10 characters).
        """
        current = lead.stage
        
        # Check if current stage can be rolled back
        if current not in REVERSIBLE_STAGE_TRANSITIONS:
            raise LeadStageError(
                f"Stage '{current.value}' cannot be rolled back. "
                f"Only {list(REVERSIBLE_STAGE_TRANSITIONS.keys())} can be rolled back."
            )
        
        # Validate reason length
        if len(reason) < 10:
            raise LeadStageError("Rollback reason must be at least 10 characters.")
        
        # Get target stage
        target_stage = REVERSIBLE_STAGE_TRANSITIONS[current]
        
        # Log history
        history = LeadHistory(
            lead_id=lead.id,
            old_stage=current.value,
            new_stage=target_stage.value,
            changed_by=changed_by,
            reason=f"ROLLBACK: {reason}"
        )
        
        self.repo.db.add(history)
        lead.stage = target_stage
        
        updated_lead = await self.repo.save(lead)
        await ws_manager.broadcast({"event": "lead_rolled_back", "lead_id": lead.id, "stage": target_stage.value})
        
        return updated_lead

    async def add_attachment(
        self, 
        lead_id: int, 
        file_name: str, 
        file_type: str, 
        file_path: str, 
        file_size: int = None,
        uploaded_by: str = None
    ) -> LeadAttachment:
        """Create a new attachment record for a lead."""
        attachment = LeadAttachment(
            lead_id=lead_id,
            file_name=file_name,
            file_type=file_type,
            file_path=file_path,
            file_size_bytes=file_size,
            uploaded_by=uploaded_by
        )
        self.repo.db.add(attachment)
        await self.repo.db.flush()
        await self.repo.db.refresh(attachment)
        return attachment

    async def get_attachments(self, lead_id: int) -> list[LeadAttachment]:
        """Fetch all attachments for a specific lead."""
        from sqlalchemy import select
        result = await self.repo.db.execute(
            select(LeadAttachment).where(LeadAttachment.lead_id == lead_id).order_by(LeadAttachment.created_at.desc())
        )
        return list(result.scalars().all())

    async def nurture_lead(self, lead: Lead, reason: str = "Stale check") -> Lead:
        """
        Log a nurture attempt and prepare for automated re-engagement.
        Does not change the stage but adds a record to history.
        """
        history = LeadHistory(
            lead_id=lead.id,
            old_stage=lead.stage.value,
            new_stage=lead.stage.value,  # Stage remains the same
            changed_by="Automation",
            reason=f"NURTURE: {reason}"
        )
        self.repo.db.add(history)
        await self.repo.db.flush()
        
        # We use history to track nurture frequency in the repository layer if needed
        return await self.repo.save(lead)
