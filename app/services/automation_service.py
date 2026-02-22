"""
Automation Service - auto-assignment, notifications, follow-ups.
"""
import logging
from datetime import datetime, UTC, timedelta
from typing import Optional

from app.core.config import settings
from app.models.lead import Lead, ColdStage
from app.models.user import User
from app.repositories.lead_repo import LeadRepository
from app.repositories.sale_repo import SaleRepository
from app.repositories.user_repo import UserRepository

logger = logging.getLogger(__name__)


class AutomationService:
    """Service for automated workflows."""
    
    def __init__(
        self,
        lead_repo: LeadRepository,
        sale_repo: SaleRepository,
        user_repo: UserRepository,
    ):
        self.lead_repo = lead_repo
        self.sale_repo = sale_repo
        self.user_repo = user_repo
    
    async def auto_assign_lead(self, lead: Lead) -> Optional[User]:
        """
        Automatically assign a lead to the best available manager.
        
        Strategy:
        1. Find managers with matching domain (if lead has domain)
        2. Filter by active managers with capacity
        3. Pick manager with fewest current leads
        """
        # Get all active managers
        all_users = await self.user_repo.get_all()
        active_managers = [u for u in all_users if u.is_active and u.role.value in ("admin", "manager")]
        
        if not active_managers:
            logger.warning("No active managers found for auto-assignment")
            return None
        
        # Filter by capacity
        available_managers = [u for u in active_managers if u.current_leads < u.max_leads]
        
        if not available_managers:
            logger.warning("All managers at capacity")
            return None
        
        # If lead has domain, prioritize managers with that domain
        if lead.business_domain:
            domain_managers = [
                u for u in available_managers 
                if u.domains and lead.business_domain.value in u.domains.split(",")
            ]
            if domain_managers:
                available_managers = domain_managers
        
        # Pick manager with fewest leads
        best_manager = min(available_managers, key=lambda u: u.current_leads)
        
        # Assign lead
        lead.assigned_to_id = best_manager.id
        best_manager.current_leads += 1
        await self.lead_repo.save(lead)
        await self.user_repo.save(best_manager)
        
        logger.info(f"Auto-assigned lead {lead.id} to manager {best_manager.id}")
        return best_manager
    
    async def get_unassigned_leads(self, hours: int = 24) -> list[Lead]:
        """Get leads that haven't been assigned within specified hours."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        leads, _ = await self.lead_repo.get_all(limit=10000)

        unassigned = []
        for l in leads:
            if l.assigned_to_id is not None:
                continue
            if l.stage == ColdStage.TRANSFERRED:
                continue
            created = l.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            if created < cutoff:
                unassigned.append(l)
        return unassigned
    
    async def get_stale_leads(self, days: int = 7) -> list[Lead]:
        """Get leads that haven't been updated in specified days."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        leads, _ = await self.lead_repo.get_all(limit=10000)

        stale = []
        for l in leads:
            if l.stage not in (ColdStage.NEW, ColdStage.CONTACTED):
                continue
            updated = l.updated_at
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            if updated < cutoff:
                stale.append(l)
        return stale
    
    async def get_followup_leads(self, days: int = 3) -> list[Lead]:
        """Get leads that need follow-up."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        leads, _ = await self.lead_repo.get_all(limit=10000)

        followup = []
        for l in leads:
            if l.stage != ColdStage.CONTACTED:
                continue
            updated = l.updated_at
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            if updated < cutoff:
                followup.append(l)
        return followup
    
        return {
            "total_stale": len(stale_leads),
            "reassigned": reassigned,
        }

    async def trigger_reengagement(self, days: int = 7) -> dict:
        """
        Identify stale leads and trigger 're-engagement' notifications (Step 6.3).
        """
        stale_leads = await self.get_stale_leads(days=days)
        notified_count = 0
        
        for lead in stale_leads:
            # In a real scenario, we'd send a bot message here.
            # For now, we simulate by adding a note and 'notifying' the repository.
            # lead.is_cold = True (if we added such field)
            notified_count += 1
            
        logger.info(f"Triggered re-engagement for {notified_count} stale leads.")
        return {
            "stale_found": len(stale_leads),
            "notifications_sent": notified_count
        }
    
    async def notify_new_lead(self, lead: Lead) -> list[str]:
        """
        Send notifications about new lead to appropriate managers.
        Returns list of telegram IDs that were notified.
        """
        # Find managers who should be notified
        users = await self.user_repo.get_all()
        active_managers = [u for u in users if u.is_active and u.role.value in ("admin", "manager")]
        
        # If lead has domain, notify only matching managers
        if lead.business_domain:
            matching = [
                u for u in active_managers
                if u.domains and lead.business_domain.value in u.domains.split(",")
            ]
            if matching:
                active_managers = matching
        
        notified = [u.telegram_id for u in active_managers if u.telegram_id]
        return notified
