"""
Celery tasks for lead processing.
"""
import logging
from datetime import datetime, timedelta, UTC

from app.celery.config import celery_app
from app.core.database import AsyncSessionLocal
from app.models.lead import Lead, ColdStage
from app.repositories.lead_repo import LeadRepository

logger = logging.getLogger(__name__)


@celery_app.task
def process_stale_leads_task() -> dict:
    """
    Process leads that haven't been contacted in configured days.
    
    Finds leads in 'new' stage older than 7 days and logs them for review.
    This is a placeholder for more complex automation.
    """
    import asyncio
    
    async def _process():
        stale_days = 7
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            
            # Find stale leads
            stale_leads = await lead_repo.get_stale_leads(days=stale_days)
            
            if stale_leads:
                logger.info(f"Found {len(stale_leads)} stale leads")
                # Here you could:
                # - Notify managers
                # - Auto-mark as lost
                # - Re-queue for analysis
                return {
                    "processed": len(stale_leads),
                    "lead_ids": [lead.id for lead in stale_leads],
                }
            
            return {"processed": 0, "lead_ids": []}
    
    return asyncio.run(_process())


@celery_app.task
def auto_followup_task(lead_id: int) -> dict:
    """
    Send followup notification for a lead.
    
    Args:
        lead_id: ID of the lead
        
    Returns:
        Dict with task result
    """
    import asyncio
    
    async def _followup():
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            
            lead = await lead_repo.get_by_id(lead_id)
            if not lead:
                return {"error": "Lead not found"}
            
            # Here you would integrate with Telegram bot
            # to send a followup message
            logger.info(f"Followup needed for lead {lead_id}: {lead.telegram_id}")
            
            return {
                "lead_id": lead_id,
                "telegram_id": lead.telegram_id,
                "status": "followup_queued",
            }
    
    return asyncio.run(_followup())


@celery_app.task
def cleanup_old_leads_task(days: int = 90) -> dict:
    """
    Archive leads that have been in 'lost' stage for more than N days.
    
    Args:
        days: Number of days after which to archive lost leads
        
    Returns:
        Dict with cleanup results
    """
    import asyncio
    
    async def _cleanup():
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            
            archived = await lead_repo.archive_old_lost_leads(days=days)
            
            logger.info(f"Archived {archived} old lost leads")
            return {"archived": archived}
    
    return asyncio.run(_cleanup())
