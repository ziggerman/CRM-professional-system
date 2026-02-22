"""
Lead Repository - Data Access Layer for Lead model.
"""
from datetime import datetime, UTC
from typing import Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.lead import Lead, ColdStage, LeadSource, BusinessDomain


class LeadRepository:
    """Repository for Lead CRUD operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(self, lead: Lead) -> Lead:
        """Create a new lead."""
        self.db.add(lead)
        await self.db.flush()
        await self.db.refresh(lead)
        return lead
    
    async def get_by_id(self, lead_id: int) -> Optional[Lead]:
        """Get lead by ID."""
        result = await self.db.execute(
            select(Lead)
            .options(selectinload(Lead.sale))
            .where(Lead.id == lead_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        stage: Optional[ColdStage] = None,
        source: Optional[str] = None,
        business_domain: Optional[str] = None,
        assigned_to_id: Optional[int] = None,
        telegram_id: Optional[str] = None,
        query: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Lead], int]:
        """Get leads with optional filtering and pagination."""
        from sqlalchemy import String, or_, func
        
        # Base statement for both count and results
        stmt = select(Lead)

        if stage:
            stmt = stmt.where(Lead.stage == stage)
        if source:
            stmt = stmt.where(Lead.source == source)
        if business_domain:
            stmt = stmt.where(Lead.business_domain == business_domain)
        if assigned_to_id is not None:
            stmt = stmt.where(Lead.assigned_to_id == assigned_to_id)
        if telegram_id:
            stmt = stmt.where(Lead.telegram_id == telegram_id)
        
        if created_after:
            stmt = stmt.where(Lead.created_at >= created_after)
        if created_before:
            stmt = stmt.where(Lead.created_at <= created_before)
            
        if query:
            q_pattern = f"%{query}%"
            stmt = stmt.where(
                or_(
                    Lead.telegram_id.ilike(q_pattern),
                    Lead.source.cast(String).ilike(q_pattern),
                    Lead.business_domain.cast(String).ilike(q_pattern)
                )
            )

        # Total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar() or 0

        # Paginated results with relationships
        stmt = stmt.options(
            selectinload(Lead.sale), 
            selectinload(Lead.notes),
            selectinload(Lead.attachments)
        )
        stmt = stmt.offset(offset).limit(limit).order_by(Lead.created_at.desc())
        
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())

        return leads, total
    
    async def save(self, lead: Lead) -> Lead:
        """Save lead changes."""
        await self.db.flush()
        await self.db.refresh(lead)
        return lead
    
    async def delete(self, lead: Lead) -> None:
        """Delete a lead."""
        await self.db.delete(lead)
        await self.db.flush()

    async def get_stale_leads(self, days: int = 7) -> list[Lead]:
        """Get leads in NEW or CONTACTED stages with no updates for N days."""
        from datetime import datetime, UTC, timedelta
        cutoff = datetime.now(UTC) - timedelta(days=days)
        
        stmt = (
            select(Lead)
            .where(Lead.stage.in_([ColdStage.NEW, ColdStage.CONTACTED]))
            .where(Lead.updated_at <= cutoff)
            .order_by(Lead.updated_at.asc())
        )
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def archive_old_lost_leads(self, days: int = 90) -> int:
        """
        Delete leads that have been in 'lost' stage for more than N days.
        Returns number of archived leads.
        """
        from datetime import datetime, UTC, timedelta
        from sqlalchemy import delete
        cutoff = datetime.now(UTC) - timedelta(days=days)
        
        stmt = (
            delete(Lead)
            .where(Lead.stage == ColdStage.LOST)
            .where(Lead.updated_at <= cutoff)
        )
        
        result = await self.db.execute(stmt)
        return result.rowcount

    async def bulk_update_stage(self, lead_ids: list[int], stage: ColdStage) -> int:
        """Bulk update stage for multiple leads (Step 6.2)."""
        stmt = select(Lead).where(Lead.id.in_(lead_ids))
        result = await self.db.execute(stmt)
        leads = result.scalars().all()
        
        for lead in leads:
            lead.stage = stage
            lead.updated_at = datetime.now(UTC)
        
        return len(leads)

    async def bulk_delete(self, lead_ids: list[int]) -> int:
        """Bulk delete multiple leads (Step 6.2)."""
        from sqlalchemy import delete
        stmt = delete(Lead).where(Lead.id.in_(lead_ids))
        result = await self.db.execute(stmt)
        return result.rowcount
