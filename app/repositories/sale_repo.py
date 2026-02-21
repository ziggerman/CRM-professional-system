"""
Sale Repository - Data Access Layer for Sale model.
"""
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.sale import Sale, SaleStage


class SaleRepository:
    """Repository for Sale CRUD operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(self, sale: Sale) -> Sale:
        """Create a new sale."""
        self.db.add(sale)
        await self.db.flush()
        await self.db.refresh(sale)
        return sale
    
    async def get_by_id(self, sale_id: int) -> Optional[Sale]:
        """Get sale by ID."""
        result = await self.db.execute(
            select(Sale)
            .options(selectinload(Sale.lead))
            .where(Sale.id == sale_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_lead_id(self, lead_id: int) -> Optional[Sale]:
        """Get sale by lead ID."""
        result = await self.db.execute(
            select(Sale)
            .options(selectinload(Sale.lead))
            .where(Sale.lead_id == lead_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(
        self, 
        stage: Optional[SaleStage] = None, 
        offset: int = 0, 
        limit: int = 50
    ) -> tuple[list[Sale], int]:
        """Get all sales with optional filtering and pagination."""
        query = select(Sale).options(selectinload(Sale.lead))
        
        if stage:
            query = query.where(Sale.stage == stage)
        
        # Get total count
        count_query = select(Sale)
        if stage:
            count_query = count_query.where(Sale.stage == stage)
        
        from sqlalchemy import func
        count_result = await self.db.execute(
            select(func.count()).select_from(count_query.subquery())
        )
        total = count_result.scalar() or 0
        
        # Get paginated results
        query = query.offset(offset).limit(limit).order_by(Sale.created_at.desc())
        result = await self.db.execute(query)
        sales = list(result.scalars().all())
        
        return sales, total
    
    async def save(self, sale: Sale) -> Sale:
        """Save sale changes."""
        await self.db.flush()
        await self.db.refresh(sale)
        return sale
    
    async def delete(self, sale: Sale) -> None:
        """Delete a sale."""
        await self.db.delete(sale)
        await self.db.flush()
