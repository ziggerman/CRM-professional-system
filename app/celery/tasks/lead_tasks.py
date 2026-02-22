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
    
    Finds leads in 'new' stage older than 7 days and notifies managers via Telegram.
    """
    import asyncio
    from aiogram import Bot
    from app.core.config import settings
    
    async def _process():
        stale_days = 7
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            
            # Find stale leads
            stale_leads = await lead_repo.get_stale_leads(days=stale_days)
            
            if stale_leads:
                logger.info(f"Found {len(stale_leads)} stale leads")
                
                # Notify managers via shared NotificationService
                from app.services.notification_service import NotificationService
                notif_svc = NotificationService()
                
                msg_text = (
                    f"🚨 <b>STALE LEADS ALERT</b> 🚨\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Found {len(stale_leads)} leads with no activity for {stale_days} days.\n\n"
                )
                for lead in stale_leads[:5]:  # Show up to 5 examples
                    name = lead.full_name or "Unnamed"
                    msg_text += f"• <b>Lead #{lead.id}</b> | {name} ({lead.stage.value})\n"
                
                if len(stale_leads) > 5:
                    msg_text += f"\n...and {len(stale_leads) - 5} more.\n"
                
                msg_text += f"\n💡 <i>Please follow up with these clients immediately.</i>"
                
                await notif_svc.notify_admins(msg_text)
                await notif_svc.close()

                # Trigger direct nurture for each lead
                for lead in stale_leads:
                    auto_followup_task.delay(lead.id)

                return {
                    "processed": len(stale_leads),
                    "lead_ids": [lead.id for lead in stale_leads],
                }
            
            return {"processed": 0, "lead_ids": []}
    
    return asyncio.run(_process())


@celery_app.task
def auto_followup_task(lead_id: int) -> dict:
    """
    Automated nurture: Send re-engagement message directly to a stalled lead.
    """
    import asyncio
    from app.services.notification_service import NotificationService
    from app.services.lead_service import LeadService
    from app.repositories.history_repo import HistoryRepository
    
    async def _followup():
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            history_repo = HistoryRepository(session)
            lead_svc = LeadService(lead_repo, history_repo)
            
            lead = await lead_svc.get_lead(lead_id)
            if not lead or not lead.telegram_id:
                return {"error": "Lead or Telegram ID not found"}
            
            # Log the nurture attempt
            await lead_svc.nurture_lead(lead, reason="Automated 7-day follow-up")
            await session.commit()
            
            # Send message via NotificationService
            notif_svc = NotificationService()
            
            nurture_text = (
                f"👋 <b>Hi {lead.full_name or ''}!</b>\n\n"
                f"We noticed we haven't heard from you in a few days regarding your interest in <b>{lead.business_domain.value if lead.business_domain else 'our services'}</b>.\n\n"
                f"Do you have any questions we can help with? We're here to assist! ✨"
            )
            
            success = await notif_svc.send_direct(lead.telegram_id, nurture_text)
            await notif_svc.close()
            
            if success:
                logger.info(f"Successfully sent nurture message to lead {lead_id}")
                return {"status": "sent", "lead_id": lead_id}
            else:
                logger.error(f"Failed to send nurture message to lead {lead_id}")
                return {"status": "failed", "lead_id": lead_id}
    
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
