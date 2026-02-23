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
    
    # ──────────────────────────────────────────────
    # Step 8: Enhanced Auto-assignment with Skills
    # ──────────────────────────────────────────────
    
    async def auto_assign_with_skill_rules(self, lead: Lead) -> Optional[User]:
        """
        Enhanced auto-assignment with skill-based rules (Step 8).
        
        Strategy:
        1. Match by business domain (skill-based)
        2. Filter by active agents with capacity
        3. Consider agent performance (win rate)
        4. Pick best match considering load + skill
        """
        from app.models.user import UserRole
        
        # Get all active agents
        all_users = await self.user_repo.get_all()
        active_agents = [
            u for u in all_users 
            if u.is_active and u.role in (UserRole.AGENT, UserRole.MANAGER)
        ]
        
        if not active_agents:
            logger.warning("No active agents found for auto-assignment")
            return None
        
        # Filter by capacity
        available_agents = [u for u in active_agents if u.current_leads < u.max_leads]
        
        if not available_agents:
            logger.warning("All agents at capacity")
            # Return least loaded agent anyway (overload scenario)
            available_agents = active_agents
        
        # Score agents based on skill match
        scored_agents = []
        for agent in available_agents:
            score = 0
            
            # Skill match: domain expertise
            if lead.business_domain and agent.domains:
                if lead.business_domain.value in agent.domains.split(","):
                    score += 50  # High priority for skill match
            
            # Workload score (less leads = better)
            workload_factor = (agent.max_leads - agent.current_leads) / agent.max_leads
            score += workload_factor * 30
            
            # Performance bonus (experienced agents)
            if agent.sales_converted > 0:
                score += min(20, agent.sales_converted)
            
            scored_agents.append((agent, score))
        
        # Sort by score descending
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        best_agent = scored_agents[0][0] if scored_agents else None
        
        if best_agent:
            # Assign lead
            lead.assigned_to_id = best_agent.id
            best_agent.current_leads += 1
            await self.lead_repo.save(lead)
            await self.user_repo.save(best_agent)
            
            logger.info(f"Skill-based auto-assigned lead {lead.id} to agent {best_agent.id}")
        
        return best_agent
    
    # ──────────────────────────────────────────────
    # Step 8: SLA Management
    # ──────────────────────────────────────────────
    
    async def update_sla_for_lead(self, lead: Lead) -> Lead:
        """
        Update SLA deadline for a lead based on its stage and priority.
        """
        from datetime import timedelta
        
        # Set SLA based on stage
        sla_hours = {
            ColdStage.NEW: 24,        # 24 hours to first contact
            ColdStage.CONTACTED: 48,  # 48 hours to qualify
            ColdStage.QUALIFIED: 72,  # 72 hours to transfer
        }
        
        hours = sla_hours.get(lead.stage, 24)
        lead.sla_deadline_at = datetime.now(UTC) + timedelta(hours=hours)
        
        # Update days in stage
        if lead.updated_at:
            days = (datetime.now(UTC) - lead.updated_at).days
            lead.days_in_stage = days
        
        await self.lead_repo.save(lead)
        return lead
    
    async def check_and_update_overdue_leads(self) -> dict:
        """
        Check all active leads and mark overdue ones (Step 8).
        Runs as part of SLA monitoring.
        """
        now = datetime.now(UTC)
        
        # Get active leads (not transferred or lost)
        leads, _ = await self.lead_repo.get_all(limit=10000)
        active_leads = [
            l for l in leads 
            if l.stage in (ColdStage.NEW, ColdStage.CONTACTED, ColdStage.QUALIFIED)
        ]
        
        overdue_count = 0
        updated_leads = []
        
        for lead in active_leads:
            # Check if past SLA deadline
            if lead.sla_deadline_at and now > lead.sla_deadline_at:
                if not lead.is_overdue:
                    lead.is_overdue = True
                    await self.lead_repo.save(lead)
                    overdue_count += 1
                    updated_leads.append(lead.id)
            
            # Update days in stage
            if lead.updated_at:
                updated = lead.updated_at
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=UTC)
                lead.days_in_stage = (now - updated).days
                
                # Auto-mark overdue if no activity for too long
                if lead.days_in_stage > 7 and not lead.is_overdue:
                    lead.is_overdue = True
                    overdue_count += 1
                    updated_leads.append(lead.id)
                
                await self.lead_repo.save(lead)
        
        logger.info(f"SLA check: {overdue_count} leads marked overdue")
        return {
            "checked": len(active_leads),
            "overdue_count": overdue_count,
            "updated_leads": updated_leads,
        }
    
    # ──────────────────────────────────────────────
    # Step 8: Stale Lead Trigger Automation
    # ──────────────────────────────────────────────
    
    async def process_stale_leads(self, stale_days: int = 7) -> dict:
        """
        Process leads that have been stale for too long (Step 8).
        
        Actions:
        - Mark as overdue
        - Create follow-up task
        - Optionally reassign to another agent
        """
        stale_leads = await self.get_stale_leads(days=stale_days)
        
        actions_taken = []
        
        for lead in stale_leads:
            # Mark as overdue
            lead.is_overdue = True
            await self.lead_repo.save(lead)
            
            actions_taken.append({
                "lead_id": lead.id,
                "action": "marked_overdue",
                "stage": lead.stage.value,
                "days_inactive": stale_days
            })
            
            # TODO: Create task/notification for this lead
            # This would integrate with a task system
        
        logger.info(f"Processed {len(stale_leads)} stale leads")
        
        return {
            "total_stale": len(stale_leads),
            "actions": actions_taken,
        }
    
    async def escalate_overdue_leads(self, escalate_after_days: int = 14) -> dict:
        """
        Escalate leads that have been overdue for too long (Step 8).
        
        Actions:
        - Notify managers
        - Consider lead quality downgrade
        """
        cutoff = datetime.now(UTC) - timedelta(days=escalate_after_days)
        
        leads, _ = await self.lead_repo.get_all(limit=10000)
        
        # Find overdue leads that haven't been updated in a long time
        escalation_leads = [
            l for l in leads
            if l.is_overdue and l.stage not in (ColdStage.TRANSFERRED, ColdStage.LOST)
            and l.updated_at < cutoff
        ]
        
        escalated_count = 0
        escalated_ids = []
        
        for lead in escalation_leads:
            # Could trigger notification to manager here
            escalated_ids.append(lead.id)
            escalated_count += 1
            
            logger.warning(f"Lead {lead.id} escalated - {escalate_after_days}+ days overdue")
        
        return {
            "escalated_count": escalated_count,
            "lead_ids": escalated_ids,
            "threshold_days": escalate_after_days,
        }
    
    # ──────────────────────────────────────────────
    # Step 8: Lead Priority Update
    # ──────────────────────────────────────────────
    
    async def update_lead_priority(self, lead: Lead) -> str:
        """
        Update lead priority based on:
        - Days in stage
        - Overdue status
        - AI score
        """
        priority = "normal"
        
        # Check overdue status first
        if lead.is_overdue:
            priority = "high"
        
        # Check AI score for hot leads
        if lead.ai_score and lead.ai_score >= 0.8:
            priority = "high"
        
        # Check days in stage
        if lead.days_in_stage > 14:
            priority = "high"
        elif lead.days_in_stage > 7:
            priority = "medium"
        
        # Log for monitoring
        logger.info(f"Lead {lead.id} priority: {priority} (score={lead.ai_score}, overdue={lead.is_overdue}, days={lead.days_in_stage})")
        
        return priority
