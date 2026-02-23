from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.history import LeadHistory


class HistoryRepository:
    """Repository for Lead history records."""

    def __init__(self, db: AsyncSession):
        self.db = db
        
    async def get_by_lead_id(self, lead_id: int) -> list[LeadHistory]:
        """Fetch all history records for a specific lead, ordered by newest first."""
        stmt = (
            select(LeadHistory)
            .where(LeadHistory.lead_id == lead_id)
            .order_by(LeadHistory.created_at.desc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
