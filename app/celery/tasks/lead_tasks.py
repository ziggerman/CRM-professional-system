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


# ──────────────────────────────────────────────
# Step 8: SLA Reminders and Escalations
# ──────────────────────────────────────────────

@celery_app.task
def sla_check_and_notify_task() -> dict:
    """
    Check SLA deadlines and send reminders for overdue leads (Step 8).
    
    Runs periodically (e.g., every hour) to:
    - Check leads approaching SLA deadline
    - Mark overdue leads
    - Send notifications to assigned agents
    """
    import asyncio
    
    async def _check():
        from app.services.automation_service import AutomationService
        from app.repositories.lead_repo import LeadRepository
        from app.repositories.user_repo import UserRepository
        from app.repositories.sale_repo import SaleRepository
        from app.services.notification_service import NotificationService
        
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            user_repo = UserRepository(session)
            sale_repo = SaleRepository(session)
            
            automation_svc = AutomationService(lead_repo, sale_repo, user_repo)
            
            # Check and update overdue leads
            result = await automation_svc.check_and_update_overdue_leads()
            
            # Get overdue leads for notification
            leads, _ = await lead_repo.get_all(limit=10000)
            overdue_leads = [
                l for l in leads 
                if l.is_overdue and l.assigned_to_id and l.stage not in (ColdStage.TRANSFERRED, ColdStage.LOST)
            ]
            
            if overdue_leads:
                # Send notification to admins
                notif_svc = NotificationService()
                
                msg = (
                    f"⚠️ <b>SLA OVERDUE ALERT</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"{len(overdue_leads)} leads are past SLA deadline.\n\n"
                )
                for lead in overdue_leads[:5]:
                    name = lead.full_name or "Unnamed"
                    msg += f"• <b>Lead #{lead.id}</b> | {name} ({lead.stage.value})\n"
                
                if len(overdue_leads) > 5:
                    msg += f"\n...and {len(overdue_leads) - 5} more.\n"
                
                await notif_svc.notify_admins(msg)
                await notif_svc.close()
            
            return result
    
    return asyncio.run(_check())


@celery_app.task
def stale_lead_automation_task(stale_days: int = 7) -> dict:
    """
    Process stale leads and trigger automation (Step 8).
    
    Actions:
    - Mark leads as overdue
    - Send notifications
    - Trigger re-engagement flow
    """
    import asyncio
    
    async def _process():
        from app.services.automation_service import AutomationService
        from app.repositories.lead_repo import LeadRepository
        from app.repositories.user_repo import UserRepository
        from app.repositories.sale_repo import SaleRepository
        from app.services.notification_service import NotificationService
        
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            user_repo = UserRepository(session)
            sale_repo = SaleRepository(session)
            
            automation_svc = AutomationService(lead_repo, sale_repo, user_repo)
            
            # Process stale leads
            result = await automation_svc.process_stale_leads(stale_days=stale_days)
            
            # Send notification if there are stale leads
            if result["total_stale"] > 0:
                notif_svc = NotificationService()
                
                msg = (
                    f"📋 <b>STALE LEADS PROCESSED</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Found {result['total_stale']} stale leads (>{stale_days} days inactive).\n\n"
                    f"Leads have been marked for follow-up."
                )
                
                await notif_svc.notify_admins(msg)
                await notif_svc.close()
            
            return result
    
    return asyncio.run(_process())


@celery_app.task
def escalate_overdue_task(escalate_after_days: int = 14) -> dict:
    """
    Escalate leads that have been overdue for too long (Step 8).
    
    Actions:
    - Notify managers of severely overdue leads
    - Log for executive review
    """
    import asyncio
    
    async def _escalate():
        from app.services.automation_service import AutomationService
        from app.repositories.lead_repo import LeadRepository
        from app.repositories.user_repo import UserRepository
        from app.repositories.sale_repo import SaleRepository
        from app.services.notification_service import NotificationService
        
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            user_repo = UserRepository(session)
            sale_repo = SaleRepository(session)
            
            automation_svc = AutomationService(lead_repo, sale_repo, user_repo)
            
            # Escalate overdue leads
            result = await automation_svc.escalate_overdue_leads(
                escalate_after_days=escalate_after_days
            )
            
            # Send urgent notification
            if result["escalated_count"] > 0:
                notif_svc = NotificationService()
                
                msg = (
                    f"🚨 <b>LEAD ESCALATION</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"{result['escalated_count']} leads have been overdue for {escalate_after_days}+ days.\n\n"
                    f"<i>Immediate attention required.</i>"
                )
                
                await notif_svc.notify_admins(msg)
                await notif_svc.close()
            
            return result
    
    return asyncio.run(_escalate())


@celery_app.task
def daily_sla_report_task() -> dict:
    """
    Generate daily SLA report for management (Step 8).
    
    Returns:
        Dict with SLA metrics for the day
    """
    import asyncio
    
    async def _report():
        from app.services.kpi_service import KPIService
        
        async with AsyncSessionLocal() as session:
            kpi_service = KPIService(session)
            
            # Get aging data
            aging = await kpi_service.get_lead_aging()
            
            # Get response time
            response = await kpi_service.get_median_response_time()
            
            # Get conversion
            conversion = await kpi_service.get_conversion_per_stage()
            
            report = {
                "generated_at": datetime.now(UTC).isoformat(),
                "sla_metrics": {
                    "overdue_leads": aging.get("overdue_leads", 0),
                    "age_distribution": aging.get("age_distribution", {}),
                },
                "response_metrics": {
                    "median_hours": response.get("median_hours"),
                    "avg_hours": response.get("avg_hours"),
                },
                "conversion": {
                    "total_leads": conversion.get("total", 0),
                    "conversion_rates": conversion.get("conversion_rates", {}),
                },
            }
            
            logger.info(f"Daily SLA report: {report}")
            return report
    
    return asyncio.run(_report())
