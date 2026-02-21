"""
Lead API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from starlette import status

from app.core.deps import get_lead_service, get_transfer_service
from app.models.lead import ColdStage
from app.schemas.lead import (
    LeadCreate,
    LeadResponse,
    LeadListResponse,
    LeadStageUpdate,
    LeadMessageUpdate,
    AIAnalysisResult,
)
from app.schemas.note import NoteCreate, NoteResponse, NoteListResponse
from app.schemas.sale import SaleResponse
from app.services.lead_service import LeadService, LeadNotFoundError, LeadStageError
from app.services.transfer_service import TransferService, TransferError
from app.ai.ai_service import AIServiceError

router = APIRouter()


def _not_found(lead_id: int):
    raise HTTPException(status_code=404, detail=f"Lead {lead_id} not found")


def _bad_request(msg: str):
    raise HTTPException(status_code=400, detail=msg)


# ──────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────

@router.post("", response_model=LeadResponse, status_code=status.HTTP_201_CREATED)
async def create_lead(
    data: LeadCreate,
    svc: LeadService = Depends(get_lead_service),
):
    """Create a new lead. Source is required; business_domain is optional."""
    lead = await svc.create_lead(data)
    return lead


@router.get("", response_model=LeadListResponse)
async def list_leads(
    stage: ColdStage | None = Query(default=None),
    source: str | None = Query(default=None, description="Filter by source"),
    business_domain: str | None = Query(default=None, description="Filter by business domain"),
    assigned_to_id: int | None = Query(default=None, description="Filter by assigned user ID"),
    telegram_id: str | None = Query(default=None, description="Filter by telegram ID"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page"),
    svc: LeadService = Depends(get_lead_service),
):
    """List leads with pagination and filters."""
    offset = (page - 1) * page_size
    items, total = await svc.get_leads(
        stage=stage, 
        source=source,
        business_domain=business_domain,
        assigned_to_id=assigned_to_id,
        telegram_id=telegram_id,
        offset=offset, 
        limit=page_size
    )
    
    total_pages = (total + page_size - 1) // page_size
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


@router.get("/{lead_id}", response_model=LeadResponse)
async def get_lead(
    lead_id: int,
    svc: LeadService = Depends(get_lead_service),
):
    try:
        return await svc.get_lead(lead_id)
    except LeadNotFoundError:
        _not_found(lead_id)


# ──────────────────────────────────────────────
# Stage management
# ──────────────────────────────────────────────

@router.patch("/{lead_id}/stage", response_model=LeadResponse)
async def update_stage(
    lead_id: int,
    data: LeadStageUpdate,
    svc: LeadService = Depends(get_lead_service),
):
    """
    Advance lead to next stage.
    Rules: sequential only, terminal stages locked.
    """
    try:
        lead = await svc.get_lead(lead_id)
        return await svc.transition_stage(lead, data.stage)
    except LeadNotFoundError:
        _not_found(lead_id)
    except LeadStageError as e:
        _bad_request(str(e))


@router.post("/{lead_id}/messages", response_model=LeadResponse)
async def record_messages(
    lead_id: int,
    data: LeadMessageUpdate,
    svc: LeadService = Depends(get_lead_service),
):
    """Increment message counter for activity tracking."""
    try:
        lead = await svc.get_lead(lead_id)
        return await svc.increment_messages(lead, data.increment)
    except LeadNotFoundError:
        _not_found(lead_id)


# ──────────────────────────────────────────────
# AI analysis
# ──────────────────────────────────────────────

@router.post("/{lead_id}/analyze", response_model=AIAnalysisResult)
async def analyze_lead(
    lead_id: int,
    force: bool = Query(default=False, description="Force re-analysis even if recent"),
    svc: LeadService = Depends(get_lead_service),
    transfer_svc: TransferService = Depends(get_transfer_service),
):
    """
    Request AI analysis of lead.
    Returns score + recommendation. Does NOT change lead stage.
    Manager reviews the recommendation and decides next action.
    
    If lead was analyzed recently (within AI_ANALYSIS_STALE_DAYS), 
    returns cached result unless force=true.
    """
    from datetime import timedelta, datetime, UTC
    from app.core.config import settings
    
    try:
        lead = await svc.get_lead(lead_id)
        
        # Check if analysis is fresh
        if not force and lead.ai_analyzed_at:
            stale_threshold = datetime.now(UTC) - timedelta(days=settings.AI_ANALYSIS_STALE_DAYS)
            # Handle naive datetime
            analyzed_at = lead.ai_analyzed_at
            if analyzed_at.tzinfo is None:
                analyzed_at = analyzed_at.replace(tzinfo=UTC)
            if analyzed_at > stale_threshold:
                # Return cached result
                return AIAnalysisResult(
                    score=lead.ai_score,
                    recommendation=lead.ai_recommendation or "",
                    reason=lead.ai_reason or "",
                )
        
        return await transfer_svc.analyze_lead(lead)
    except LeadNotFoundError:
        _not_found(lead_id)
    except AIServiceError as e:
        raise HTTPException(status_code=503, detail=f"AI service error: {e}")


# ──────────────────────────────────────────────
# Transfer to sales
# ──────────────────────────────────────────────

@router.post("/{lead_id}/transfer", response_model=SaleResponse, status_code=status.HTTP_201_CREATED)
async def transfer_lead(
    lead_id: int,
    amount: int = Query(default=None, ge=0, description="Initial deal amount (optional)"),
    svc: LeadService = Depends(get_lead_service),
    transfer_svc: TransferService = Depends(get_transfer_service),
):
    """
    Transfer qualified lead to sales pipeline.

    Requirements (enforced server-side):
    - AI score >= 0.6
    - Business domain must be set
    - Lead must not be already transferred

    This endpoint is intentionally separate from stage update —
    it's a significant business action with multiple gates.
    """
    try:
        lead = await svc.get_lead(lead_id)
        _, sale = await transfer_svc.transfer_to_sales(lead, amount)
        return sale
    except LeadNotFoundError:
        _not_found(lead_id)
    except TransferError as e:
        _bad_request(str(e))


# ──────────────────────────────────────────────
# Notes/Comments
# ──────────────────────────────────────────────

@router.get("/{lead_id}/notes", response_model=NoteListResponse)
async def list_lead_notes(
    lead_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    svc: LeadService = Depends(get_lead_service),
):
    """Get all notes for a lead."""
    lead = await svc.get_lead(lead_id)
    
    notes = lead.notes
    total = len(notes)
    
    # Manual pagination
    start = (page - 1) * page_size
    end = start + page_size
    page_notes = notes[start:end]
    
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    
    return {
        "items": page_notes,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


@router.post("/{lead_id}/notes", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
async def create_lead_note(
    lead_id: int,
    data: NoteCreate,
    svc: LeadService = Depends(get_lead_service),
):
    """Add a note to a lead."""
    from datetime import datetime, UTC
    from app.models.note import LeadNote
    
    lead = await svc.get_lead(lead_id)
    
    note = LeadNote(
        lead_id=lead.id,
        content=data.content,
        note_type=data.note_type,
        author_id=data.author_id,
        author_name=data.author_name,
    )
    
    # Save note
    from app.core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        session.add(note)
        await session.commit()
        await session.refresh(note)
    
    return note


@router.delete("/{lead_id}/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lead_note(
    lead_id: int,
    note_id: int,
    svc: LeadService = Depends(get_lead_service),
):
    """Delete a note from a lead."""
    from app.core.database import AsyncSessionLocal
    from sqlalchemy import select
    from app.models.note import LeadNote
    
    # Verify lead exists
    await svc.get_lead(lead_id)
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(LeadNote).where(LeadNote.id == note_id, LeadNote.lead_id == lead_id)
        )
        note = result.scalar_one_or_none()
        
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        
        await session.delete(note)
        await session.commit()
    
    return None


# ──────────────────────────────────────────────
# Lead Assignment
# ──────────────────────────────────────────────

@router.post("/{lead_id}/assign/{user_id}", response_model=LeadResponse)
async def assign_lead(
    lead_id: int,
    user_id: int,
    svc: LeadService = Depends(get_lead_service),
):
    """Assign lead to a user/manager."""
    from app.core.database import AsyncSessionLocal
    from app.models.user import User
    from sqlalchemy import select
    
    lead = await svc.get_lead(lead_id)
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.is_active:
            raise HTTPException(status_code=400, detail="User is not active")
        
        # Check if user has capacity
        if user.current_leads >= user.max_leads:
            raise HTTPException(status_code=400, detail="User has reached max leads capacity")
        
        lead.assigned_to_id = user_id
        user.current_leads += 1
        
        await session.commit()
        await session.refresh(lead)
    
    return lead


@router.post("/{lead_id}/unassign", response_model=LeadResponse)
async def unassign_lead(
    lead_id: int,
    svc: LeadService = Depends(get_lead_service),
):
    """Unassign lead from user."""
    from app.core.database import AsyncSessionLocal
    from app.models.user import User
    from sqlalchemy import select
    
    lead = await svc.get_lead(lead_id)
    
    if lead.assigned_to_id:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(User).where(User.id == lead.assigned_to_id))
            user = result.scalar_one_or_none()
            
            if user and user.current_leads > 0:
                user.current_leads -= 1
            
            lead.assigned_to_id = None
            await session.commit()
            await session.refresh(lead)
    
    return lead


@router.post("/{lead_id}/reassign/{new_user_id}", response_model=LeadResponse)
async def reassign_lead(
    lead_id: int,
    new_user_id: int,
    svc: LeadService = Depends(get_lead_service),
):
    """
    Reassign lead from current user to another user.
    This is used to transfer lead ownership between managers.
    """
    from app.core.database import AsyncSessionLocal
    from app.models.user import User
    from sqlalchemy import select
    
    lead = await svc.get_lead(lead_id)
    old_user_id = lead.assigned_to_id
    
    async with AsyncSessionLocal() as session:
        # Get new user
        result = await session.execute(select(User).where(User.id == new_user_id))
        new_user = result.scalar_one_or_none()
        
        if not new_user:
            raise HTTPException(status_code=404, detail="New user not found")
        
        if not new_user.is_active:
            raise HTTPException(status_code=400, detail="New user is not active")
        
        if new_user.current_leads >= new_user.max_leads:
            raise HTTPException(status_code=400, detail="New user has reached max leads capacity")
        
        # Decrease old user's count
        if old_user_id:
            result = await session.execute(select(User).where(User.id == old_user_id))
            old_user = result.scalar_one_or_none()
            if old_user and old_user.current_leads > 0:
                old_user.current_leads -= 1
        
        # Increase new user's count
        new_user.current_leads += 1
        
        # Assign lead
        lead.assigned_to_id = new_user_id
        
        await session.commit()
        await session.refresh(lead)
    
    return lead
