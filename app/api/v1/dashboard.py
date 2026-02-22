"""
Dashboard API - statistics and metrics.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.deps import get_automation_service
from app.services.automation_service import AutomationService
from app.core.config import settings
import json
import redis.asyncio as redis

from app.core.security import verify_api_token, require_role

router = APIRouter(
    prefix="/api/v1/dashboard", 
    tags=["dashboard"],
    dependencies=[Depends(require_role("admin"))]
)


class LeadStats(BaseModel):
    """Lead statistics."""
    total: int
    new: int
    contacted: int
    qualified: int
    transferred: int
    lost: int


class SalesStats(BaseModel):
    """Sales statistics."""
    total: int
    new: int
    kyc: int
    agreement: int
    paid: int
    lost: int


class DashboardStats(BaseModel):
    """Complete dashboard statistics."""
    leads: LeadStats
    sales: SalesStats
    conversion_rate: float
    avg_deal_amount: float | None
    total_revenue: int


@router.get("", response_model=DashboardStats)
async def get_dashboard():
    """Get complete dashboard statistics with Redis caching (Step 5.2)."""
    cache_key = "dashboard_stats_main"
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)
    
    # Try cache
    try:
        cached = await r.get(cache_key)
        if cached:
            return DashboardStats(**json.loads(cached))
    except Exception:
        pass

    from app.repositories.lead_repo import LeadRepository
    from app.repositories.sale_repo import SaleRepository
    from app.core.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as session:
        lead_repo = LeadRepository(session)
        sale_repo = SaleRepository(session)
        
        # Get all leads and sales (expensive)
        leads, _ = await lead_repo.get_all()
        sales, _ = await sale_repo.get_all()
        
        # Count leads by stage
        lead_counts = {
            "total": len(leads),
            "new": sum(1 for l in leads if l.stage.value == "new"),
            "contacted": sum(1 for l in leads if l.stage.value == "contacted"),
            "qualified": sum(1 for l in leads if l.stage.value == "qualified"),
            "transferred": sum(1 for l in leads if l.stage.value == "transferred"),
            "lost": sum(1 for l in leads if l.stage.value == "lost"),
        }
        
        # Count sales by stage
        sales_counts = {
            "total": len(sales),
            "new": sum(1 for s in sales if s.stage.value == "new"),
            "kyc": sum(1 for s in sales if s.stage.value == "kyc"),
            "agreement": sum(1 for s in sales if s.stage.value == "agreement"),
            "paid": sum(1 for s in sales if s.stage.value == "paid"),
            "lost": sum(1 for s in sales if s.stage.value == "lost"),
        }
        
        # Calculate conversion rate
        transferred = lead_counts["transferred"]
        total = lead_counts["total"]
        conversion_rate = (transferred / total * 100) if total > 0 else 0
        
        # Calculate revenue
        paid_sales = [s for s in sales if s.stage.value == "paid"]
        total_revenue = sum(s.amount or 0 for s in paid_sales)
        avg_deal = total_revenue / len(paid_sales) if paid_sales else None
        
        result = DashboardStats(
            leads=LeadStats(**lead_counts),
            sales=SalesStats(**sales_counts),
            conversion_rate=round(conversion_rate, 2),
            avg_deal_amount=avg_deal,
            total_revenue=total_revenue,
        )

        # Update cache (5 mins)
        try:
            await r.setex(cache_key, 300, result.model_dump_json())
        except Exception:
            pass
        finally:
            await r.close()
            
        return result


@router.get("/leads-by-stage")
async def get_leads_by_stage():
    """Get leads grouped by stage for chart."""
    from app.repositories.lead_repo import LeadRepository
    from app.core.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as session:
        lead_repo = LeadRepository(session)
        leads, _ = await lead_repo.get_all()
        
        stages = {
            "new": 0,
            "contacted": 0,
            "qualified": 0,
            "transferred": 0,
            "lost": 0,
        }
        
        for lead in leads:
            if lead.stage.value in stages:
                stages[lead.stage.value] += 1
        
        return {
            "labels": list(stages.keys()),
            "values": list(stages.values()),
        }


@router.get("/sales-by-stage")
async def get_sales_by_stage():
    """Get sales grouped by stage for chart."""
    from app.repositories.sale_repo import SaleRepository
    from app.core.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as session:
        sale_repo = SaleRepository(session)
        sales, _ = await sale_repo.get_all()
        
        stages = {
            "new": 0,
            "kyc": 0,
            "agreement": 0,
            "paid": 0,
            "lost": 0,
        }
        
        for sale in sales:
            if sale.stage.value in stages:
                stages[sale.stage.value] += 1
        
        return {
            "labels": list(stages.keys()),
            "values": list(stages.values()),
        }


@router.get("/conversion-funnel")
async def get_conversion_funnel():
    """Get conversion funnel data."""
    from app.repositories.lead_repo import LeadRepository
    from app.repositories.sale_repo import SaleRepository
    from app.core.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as session:
        lead_repo = LeadRepository(session)
        sale_repo = SaleRepository(session)
        
        leads, _ = await lead_repo.get_all()
        sales, _ = await sale_repo.get_all()
        
        # Funnel stages
        new_leads = sum(1 for l in leads if l.stage.value == "new")
        contacted = sum(1 for l in leads if l.stage.value in ("new", "contacted"))
        qualified = sum(1 for l in leads if l.stage.value in ("new", "contacted", "qualified"))
        transferred = sum(1 for l in leads if l.stage.value == "transferred")
        
        # Sales funnel
        sales_started = sum(1 for s in sales if s.stage.value in ("new", "kyc"))
        sales_agreement = sum(1 for s in sales if s.stage.value in ("new", "kyc", "agreement"))
        sales_paid = sum(1 for s in sales if s.stage.value == "paid")
        
        return {
            "leads_funnel": {
                "labels": ["New", "Contacted", "Qualified", "Transferred"],
                "values": [new_leads, contacted, qualified, transferred],
            },
            "sales_funnel": {
                "labels": ["Started", "KYC", "Agreement", "Paid"],
                "values": [len(sales), sales_started, sales_agreement, sales_paid],
            },
        }


@router.get("/revenue-by-month")
async def get_revenue_by_month():
    """Get revenue grouped by month."""
    from app.repositories.sale_repo import SaleRepository
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        sale_repo = SaleRepository(session)
        sales, _ = await sale_repo.get_all(limit=10000)  # unpack tuple

        # Group by month
        monthly = {}
        for sale in sales:
            if sale.stage.value == "paid" and sale.amount:
                month = sale.created_at.strftime("%Y-%m")
                monthly[month] = monthly.get(month, 0) + sale.amount

        # Sort months chronologically
        sorted_months = sorted(monthly.items())
        return {
            "labels": [m for m, _ in sorted_months],
            "values": [v for _, v in sorted_months],
        }
@router.get("/advanced")
async def get_advanced_dashboard():
    """Get advanced analytical report."""
    from app.celery.tasks.statistics_tasks import generate_advanced_report_task
    # We call the function logic directly for the API to ensure immediate response
    # or we could run it as a task, but for a bot's immediate view, direct is better.
    return generate_advanced_report_task()
