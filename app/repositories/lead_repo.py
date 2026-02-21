"""
Lead Repository - Data Access Layer for Lead model.
"""
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.lead import Lead, ColdStage


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
        offset: int = 0, 
        limit: int = 50
    ) -> tuple[list[Lead], int]:
        """Get all leads with optional filtering and pagination."""
        query = select(Lead).options(selectinload(Lead.sale))
        
        if stage:
            query = query.where(Lead.stage == stage)
        
        # Get total count
        count_query = select(Lead)
        if stage:
            count_query = count_query.where(Lead.stage == stage)
        
        from sqlalchemy import func
        count_result = await self.db.execute(
            select(func.count()).select_from(count_query.subquery())
        )
        total = count_result.scalar() or 0
        
        # Get paginated results
        query = query.offset(offset).limit(limit).order_by(Lead.created_at.desc())
        result = await self.db.execute(query)
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
