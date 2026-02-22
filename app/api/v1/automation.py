"""
Automation API endpoints.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.deps import get_automation_service
from app.services.automation_service import AutomationService


from app.core.security import get_current_user, require_role
from app.models.user import User


router = APIRouter(
    prefix="/api/v1/automation", 
    tags=["automation"],
    dependencies=[Depends(get_current_user)]
)


class AutoAssignRequest(BaseModel):
    """Request to auto-assign a lead."""
    lead_id: int


class AutoAssignResponse(BaseModel):
    """Response for auto-assign."""
    success: bool
    lead_id: int
    assigned_to: int | None
    message: str


class ProcessStaleResponse(BaseModel):
    """Response for stale lead processing."""
    total_stale: int
    reassigned: int


@router.post("/assign/{lead_id}", response_model=AutoAssignResponse)
async def auto_assign_lead(
    lead_id: int,
    svc: AutomationService = Depends(get_automation_service),
):
    """Auto-assign a lead to the best available manager."""
    from app.repositories.lead_repo import LeadRepository
    from app.core.database import AsyncSessionLocal
    from sqlalchemy import select
    from app.models.lead import Lead
    
    async with AsyncSessionLocal() as session:
        lead_repo = LeadRepository(session)
        lead = await lead_repo.get_by_id(lead_id)
        
        if not lead:
            return AutoAssignResponse(
                success=False,
                lead_id=lead_id,
                assigned_to=None,
                message="Lead not found"
            )
        
        manager = await svc.auto_assign_lead(lead)
        
        return AutoAssignResponse(
            success=True,
            lead_id=lead_id,
            assigned_to=manager.id if manager else None,
            message=f"Assigned to {manager.full_name}" if manager else "No available manager"
        )


@router.get("/stale", response_model=ProcessStaleResponse)
async def get_stale_leads(
    days: int = 7,
    svc: AutomationService = Depends(get_automation_service),
):
    """Get leads that haven't been updated in specified days."""
    stale = await svc.get_stale_leads(days=days)
    
    return ProcessStaleResponse(
        total_stale=len(stale),
        reassigned=0,
    )


@router.post("/process-stale", response_model=ProcessStaleResponse)
async def process_stale_leads(
    svc: AutomationService = Depends(get_automation_service),
):
    """Process stale leads - try to reassign."""
    result = await svc.process_stale_leads()
    
    return ProcessStaleResponse(**result)


@router.get("/followup")
async def get_followup_leads(
    days: int = 3,
    svc: AutomationService = Depends(get_automation_service),
):
    """Get leads that need follow-up."""
    leads = await svc.get_followup_leads(days=days)
    
    return {
        "total": len(leads),
        "leads": [{"id": l.id, "stage": l.stage.value} for l in leads]
    }


@router.get("/unassigned")
async def get_unassigned_leads(
    hours: int = 24,
    svc: AutomationService = Depends(get_automation_service),
):
    """Get unassigned leads older than specified hours."""
    leads = await svc.get_unassigned_leads(hours=hours)
    
    return {
        "total": len(leads),
        "leads": [{"id": l.id, "created_at": l.created_at.isoformat()} for l in leads]
    }


@router.post("/reengage")
async def trigger_reengagement(
    days: int = 7,
    svc: AutomationService = Depends(get_automation_service),
):
    """
    Trigger re-engagement workflows for stale leads.
    Step 6.3 — Business Logic
    """
    return await svc.trigger_reengagement(days=days)
