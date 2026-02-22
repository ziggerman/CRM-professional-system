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
    Calculates:
    - Total leads by stage
    - Total sales by stage
    - Conversion rate
    - Average AI score
    - NEW: Data Capture Depth (Phone, Email, Name)
    - NEW: Intent Distribution
    """
    import asyncio
    
    async def _generate():
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            sale_repo = SaleRepository(session)
            
            all_leads, _ = await lead_repo.get_all(limit=50000)
            all_sales, _ = await sale_repo.get_all(limit=50000)
            
            total_leads = len(all_leads)
            
            # Counts by stage
            leads_by_stage = {}
            for l in all_leads:
                s = l.stage.value
                leads_by_stage[s] = leads_by_stage.get(s, 0) + 1
            
            sales_by_stage = {}
            for s in all_sales:
                st = s.stage.value
                sales_by_stage[st] = sales_by_stage.get(st, 0) + 1
            
            # Depth metrics
            has_phone = len([l for l in all_leads if l.phone])
            has_email = len([l for l in all_leads if l.email])
            has_name = len([l for l in all_leads if l.full_name])
            
            # B2B depth
            has_b2b = len([l for l in all_leads if l.company or l.budget])
            
            # Intent metrics
            intent_dist = {}
            for l in all_leads:
                if l.intent:
                    intent_dist[l.intent] = intent_dist.get(l.intent, 0) + 1
            
            # AI score
            scored_leads = [l for l in all_leads if l.ai_score is not None]
            avg_ai_score = sum(l.ai_score for l in scored_leads) / len(scored_leads) if scored_leads else 0
            
            stats = {
                "generated_at": datetime.now(UTC).isoformat(),
                "leads": {
                    "total": total_leads,
                    "by_stage": leads_by_stage,
                },
                "sales": {
                    "total": len(all_sales),
                    "by_stage": sales_by_stage,
                },
                "depth_metrics": {
                    "name_capture": round(has_name / total_leads * 100, 1) if total_leads > 0 else 0,
                    "phone_capture": round(has_phone / total_leads * 100, 1) if total_leads > 0 else 0,
                    "email_capture": round(has_email / total_leads * 100, 1) if total_leads > 0 else 0,
                    "b2b_coverage": round(has_b2b / total_leads * 100, 1) if total_leads > 0 else 0,
                },
                "intent_metrics": intent_dist,
                "metrics": {
                    "conversion_rate": round((leads_by_stage.get(ColdStage.TRANSFERRED.value, 0) / total_leads * 100), 2) if total_leads > 0 else 0,
                    "avg_ai_score": round(avg_ai_score, 2),
                },
            }
            
            logger.info(f"Advanced daily statistics generated.")
            return stats
    
    return asyncio.run(_generate())


@celery_app.task
def generate_advanced_report_task() -> dict:
    """
    Generate an advanced analytical report for admin view.
    Includes Intent distribution and B2B coverage.
    """
    import asyncio
    
    async def _generate():
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            all_leads, _ = await lead_repo.get_all(limit=50000)
            
            total = len(all_leads)
            if total == 0:
                return {"total": 0, "error": "No leads found"}
            
            # Intents
            intents = {}
            for l in all_leads:
                if l.intent:
                    intents[l.intent] = intents.get(l.intent, 0) + 1
            
            # B2B Depth
            has_company = len([l for l in all_leads if l.company])
            has_budget = len([l for l in all_leads if l.budget])
            has_pain = len([l for l in all_leads if l.pain_points])
            
            # Contacts
            has_email = len([l for l in all_leads if l.email])
            has_phone = len([l for l in all_leads if l.phone])
            
            return {
                "generated_at": datetime.now(UTC).isoformat(),
                "total_leads": total,
                "intents": intents,
                "coverage": {
                    "email": round(has_email / total * 100, 1),
                    "phone": round(has_phone / total * 100, 1),
                    "b2b_company": round(has_company / total * 100, 1),
                    "b2b_budget": round(has_budget / total * 100, 1),
                    "b2b_pain": round(has_pain / total * 100, 1),
                }
            }
            
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
