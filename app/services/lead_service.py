"""
LeadService — owns all business rules for lead lifecycle.

Rules enforced here (NOT in AI, NOT in API layer):
- Stage transitions must be sequential (no skipping)
- Terminal stages (transferred, lost) cannot be changed
- message_count can only increment
"""
from datetime import datetime, UTC

from app.models.lead import Lead, ColdStage, COLD_STAGE_ORDER, TERMINAL_COLD_STAGES
from app.models.history import LeadHistory
from app.repositories.lead_repo import LeadRepository
from app.repositories.history_repo import HistoryRepository
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


class LeadService:
    def __init__(self, lead_repo: LeadRepository, history_repo: "HistoryRepository"):
        self.repo = lead_repo
        self.history_repo = history_repo

    async def create_lead(self, data: LeadCreate) -> Lead:
        # Step 1.7: Duplicate check by email and phone
        if data.email or data.phone:
            from sqlalchemy import select, or_
            filters = []
            if data.email:
                filters.append(Lead.email == data.email)
            if data.phone:
                filters.append(Lead.phone == data.phone)
            result = await self.repo.db.execute(
                select(Lead.id, Lead.email, Lead.phone)
                .where(or_(*filters))
                .limit(1)
            )
            existing = result.first()
            if existing:
                field = "email" if (data.email and existing.email == data.email) else "phone"
                raise DuplicateLeadError(existing_id=existing.id, field=field)

        lead = Lead(
            source=data.source,
            business_domain=data.business_domain,
            telegram_id=data.telegram_id,
            full_name=data.full_name,
            email=data.email,
            phone=data.phone,
            external_username=data.external_username,
            intent=data.intent,
            company=data.company,
            position=data.position,
            budget=data.budget,
            pain_points=data.pain_points,
            stage=ColdStage.NEW,
            message_count=0,
        )
        lead = await self.repo.create(lead)
        
        # Broadcast update (Step 8.2)
        await ws_manager.broadcast({"type": "LEAD_CREATED", "id": lead.id, "stage": lead.stage.value})
        
        return lead

    async def get_lead(self, lead_id: int) -> Lead:
        lead = await self.repo.get_by_id(lead_id)
        if not lead:
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
        self, lead: Lead, new_stage: ColdStage, changed_by: str = "System"
    ) -> Lead:
        """
        Advance lead to next stage with strict validation.
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

        # Log history before saving
        history = LeadHistory(
            lead_id=lead.id,
            old_stage=current.value,
            new_stage=new_stage.value,
            changed_by=changed_by,
            reason=f"Transitioned to {new_stage.value}"
        )
        
        # Save both inside the same transaction
        self.repo.db.add(history)
        lead.stage = new_stage
        updated_lead = await self.repo.save(lead)
        await ws_manager.broadcast({"event": "lead_updated", "lead_id": lead.id, "stage": new_stage.value})
        return updated_lead

    async def increment_messages(self, lead: Lead, count: int = 1) -> Lead:
        lead.message_count += count
        return await self.repo.save(lead)

    async def update_lead(self, lead: Lead, data: "LeadUpdate") -> Lead:
        """Update lead details."""
        if data.source is not None:
            lead.source = data.source
        if data.business_domain is not None:
            lead.business_domain = data.business_domain
        if data.full_name is not None:
            lead.full_name = data.full_name
        if data.email is not None:
            lead.email = data.email
        if data.phone is not None:
            lead.phone = data.phone
        if data.external_username is not None:
            lead.external_username = data.external_username
        if data.intent is not None:
            lead.intent = data.intent
        if data.company is not None:
            lead.company = data.company
        if data.position is not None:
            lead.position = data.position
        if data.budget is not None:
            lead.budget = data.budget
        if data.pain_points is not None:
            lead.pain_points = data.pain_points
            
        return await self.repo.save(lead)

    async def save_ai_analysis(self, lead: Lead, result: AIAnalysisResult) -> Lead:
        """Persist AI result to lead. Does NOT trigger any state change."""
        lead.ai_score = result.score
        lead.ai_recommendation = result.recommendation
        lead.ai_reason = result.reason
        lead.ai_analyzed_at = datetime.now(UTC)
        return await self.repo.save(lead)

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
