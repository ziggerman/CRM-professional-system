"""
Celery tasks for statistics and reporting.
"""
import logging
from datetime import datetime, timedelta, UTC
from typing import Any

from app.celery.config import celery_app
from app.core.database import AsyncSessionLocal
from app.models.lead import ColdStage
from app.models.sale import SaleStage
from app.repositories.lead_repo import LeadRepository
from app.repositories.sale_repo import SaleRepository

logger = logging.getLogger(__name__)


@celery_app.task
def generate_daily_statistics_task() -> dict:
    """
    Generate daily statistics for leads and sales.
    
    Runs daily at 8 AM UTC (configured in celery config).
    Calculates:
    - Total leads by stage
    - Total sales by stage
    - Conversion rate
    - Average AI score
    """
    import asyncio
    
    async def _generate():
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            sale_repo = SaleRepository(session)
            
            # Get counts by stage
            leads_by_stage = {}
            for stage in ColdStage:
                leads, _ = await lead_repo.get_all(stage=stage, limit=10000)
                leads_by_stage[stage.value] = len(leads)
            
            sales_by_stage = {}
            for stage in SaleStage:
                sales, _ = await sale_repo.get_all(stage=stage, limit=10000)
                sales_by_stage[stage.value] = len(sales)
            
            # Calculate conversion rate
            total_leads = sum(leads_by_stage.values())
            transferred = leads_by_stage.get(ColdStage.TRANSFERRED.value, 0)
            conversion_rate = (transferred / total_leads * 100) if total_leads > 0 else 0
            
            # Get leads with AI scores
            all_leads, _ = await lead_repo.get_all(limit=10000)
            scored_leads = [l for l in all_leads if l.ai_score is not None]
            avg_ai_score = (
                sum(l.ai_score for l in scored_leads) / len(scored_leads)
                if scored_leads else 0
            )
            
            stats = {
                "generated_at": datetime.now(UTC).isoformat(),
                "leads": {
                    "total": total_leads,
                    "by_stage": leads_by_stage,
                },
                "sales": {
                    "total": sum(sales_by_stage.values()),
                    "by_stage": sales_by_stage,
                },
                "metrics": {
                    "conversion_rate": round(conversion_rate, 2),
                    "avg_ai_score": round(avg_ai_score, 2),
                    "analyzed_leads": len(scored_leads),
                },
            }
            
            logger.info(f"Daily statistics generated: {stats}")
            
            # Here you could:
            # - Save to database
            # - Send to Telegram
            # - Push to analytics service
            
            return stats
    
    return asyncio.run(_generate())


@celery_app.task
def generate_lead_report_task(start_date: str, end_date: str) -> dict:
    """
    Generate a report for a date range.
    
    Args:
        start_date: Start date in ISO format (YYYY-MM-DD)
        end_date: End date in ISO format (YYYY-MM-DD)
        
    Returns:
        Dict with report data
    """
    import asyncio
    
    async def _generate():
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            sale_repo = SaleRepository(session)
            
            # Get all leads created in date range
            all_leads, _ = await lead_repo.get_all(limit=10000)
            date_filtered = [
                l for l in all_leads 
                if start <= l.created_at.replace(tzinfo=UTC) <= end
            ]
            
            # Calculate metrics
            new_leads = len([l for l in date_filtered if l.stage == ColdStage.NEW])
            converted = len([l for l in date_filtered if l.stage == ColdStage.TRANSFERRED])
            lost = len([l for l in date_filtered if l.stage == ColdStage.LOST])
            
            return {
                "period": {"start": start_date, "end": end_date},
                "leads": {
                    "total": len(date_filtered),
                    "new": new_leads,
                    "converted": converted,
                    "lost": lost,
                    "conversion_rate": round(converted / len(date_filtered) * 100, 2) if date_filtered else 0,
                },
            }
    
    return asyncio.run(_generate())


@celery_app.task
def export_leads_task(filters: dict | None = None) -> dict:
    """
    Export leads with optional filters.
    
    Args:
        filters: Optional filters (stage, source, date_from, date_to)
        
    Returns:
        Dict with export status
    """
    import asyncio
    import csv
    import io
    
    async def _export():
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            
            # Get leads
            leads, _ = await lead_repo.get_all(limit=10000)
            
            # Convert to CSV
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "ID", "Source", "Stage", "Business Domain", 
                "Message Count", "AI Score", "Created At"
            ])
            
            for lead in leads:
                writer.writerow([
                    lead.id,
                    lead.source.value,
                    lead.stage.value,
                    lead.business_domain.value if lead.business_domain else "",
                    lead.message_count,
                    lead.ai_score or "",
                    lead.created_at.isoformat(),
                ])
            
            # Here you would:
            # - Upload to S3/GCS
            # - Send via email
            # - Save to file storage
            
            logger.info(f"Exported {len(leads)} leads")
            
            return {
                "exported": len(leads),
                "filename": f"leads_export_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv",
            }
    
    return asyncio.run(_export())
