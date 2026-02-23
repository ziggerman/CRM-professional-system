"""
Sale Repository - Data Access Layer for Sale model.
"""
from datetime import datetime, UTC
from typing import Optional
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.sale import Sale, SaleStage
from app.models.lead import Lead
from app.models.user import User


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

    async def get_sales_analytics(self) -> dict:
        """Aggregate sales analytics without loading all rows into memory."""
        stage_cases = {
            "new": func.sum(case((Sale.stage == SaleStage.NEW, 1), else_=0)),
            "kyc": func.sum(case((Sale.stage == SaleStage.KYC, 1), else_=0)),
            "agreement": func.sum(case((Sale.stage == SaleStage.AGREEMENT, 1), else_=0)),
            "paid": func.sum(case((Sale.stage == SaleStage.PAID, 1), else_=0)),
            "lost": func.sum(case((Sale.stage == SaleStage.LOST, 1), else_=0)),
        }

        amount_cases = {
            "agreement_value": func.sum(case((Sale.stage == SaleStage.AGREEMENT, func.coalesce(Sale.amount, 0)), else_=0)),
            "kyc_value": func.sum(case((Sale.stage == SaleStage.KYC, func.coalesce(Sale.amount, 0)), else_=0)),
            "paid_revenue": func.sum(case((Sale.stage == SaleStage.PAID, func.coalesce(Sale.amount, 0)), else_=0)),
        }

        totals_stmt = select(
            func.count(Sale.id),
            *stage_cases.values(),
            *amount_cases.values(),
        )
        totals_row = (await self.db.execute(totals_stmt)).one()

        total_sales = totals_row[0] or 0
        stage_counts = {
            "new": totals_row[1] or 0,
            "kyc": totals_row[2] or 0,
            "agreement": totals_row[3] or 0,
            "paid": totals_row[4] or 0,
            "lost": totals_row[5] or 0,
        }
        agreement_value = totals_row[6] or 0
        kyc_value = totals_row[7] or 0
        paid_revenue = totals_row[8] or 0

        paid_conversion_rate = round((stage_counts["paid"] / total_sales * 100.0), 2) if total_sales else 0.0
        weighted_forecast = round((agreement_value * 0.6) + (kyc_value * 0.3), 2)

        top_stmt = (
            select(
                User.id,
                User.full_name,
                func.count(Sale.id).label("paid_deals"),
                func.coalesce(func.sum(Sale.amount), 0).label("paid_revenue"),
            )
            .join(Lead, Lead.assigned_to_id == User.id)
            .join(Sale, Sale.lead_id == Lead.id)
            .where(Sale.stage == SaleStage.PAID)
            .group_by(User.id, User.full_name)
            .order_by(func.count(Sale.id).desc(), func.coalesce(func.sum(Sale.amount), 0).desc())
            .limit(5)
        )
        top_rows = (await self.db.execute(top_stmt)).all()

        return {
            "total_sales": total_sales,
            "paid_sales": stage_counts["paid"],
            "lost_sales": stage_counts["lost"],
            "paid_conversion_rate": paid_conversion_rate,
            "total_revenue": paid_revenue,
            "agreement_pipeline_value": agreement_value,
            "kyc_pipeline_value": kyc_value,
            "weighted_forecast_revenue": weighted_forecast,
            "funnel": [
                {"stage": SaleStage.NEW, "count": stage_counts["new"]},
                {"stage": SaleStage.KYC, "count": stage_counts["kyc"]},
                {"stage": SaleStage.AGREEMENT, "count": stage_counts["agreement"]},
                {"stage": SaleStage.PAID, "count": stage_counts["paid"]},
                {"stage": SaleStage.LOST, "count": stage_counts["lost"]},
            ],
            "top_managers": [
                {
                    "manager_id": row[0],
                    "manager_name": row[1],
                    "paid_deals": row[2] or 0,
                    "paid_revenue": row[3] or 0,
                }
                for row in top_rows
            ],
        }
