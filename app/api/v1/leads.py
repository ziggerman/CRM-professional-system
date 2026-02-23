"""
Lead API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Header, Request
from pydantic import BaseModel, Field
from typing import List, Optional
from starlette import status
import os
import aiofiles
from datetime import datetime, UTC
from pathlib import Path


from app.core.deps import get_lead_service, get_transfer_service, get_history_repo
from app.repositories.history_repo import HistoryRepository
from app.services.lead_service import LeadService, LeadNotFoundError, LeadStageError, DuplicateLeadError, MandatoryFieldsError
from app.services.transfer_service import TransferService, TransferError
from app.ai.ai_service import AIServiceError
from app.models.lead import ColdStage, LeadSource, BusinessDomain
from app.api.errors import raise_api_error
from app.core.idempotency import idempotency_store
from app.models.user import UserRole
import base64
import json

from app.schemas.lead import (
    LeadCreate,
    LeadResponse,
    LeadListResponse,
    CursorPageResponse,
    LeadStageUpdate,
    LeadUpdate,
    LeadMessageUpdate,
    AIAnalysisResult,
    LeadHistoryResponse,
    LeadAttachmentResponse,
)
from app.schemas.note import NoteCreate, NoteResponse, NoteListResponse
from app.schemas.sale import SaleResponse
from app.core.security import get_current_user, require_role
from app.models.user import User


router = APIRouter(dependencies=[Depends(get_current_user)])


def _not_found(lead_id: int):
    raise_api_error(
        status_code=404,
        code="lead_not_found",
        message="Lead not found",
        detail=f"Lead {lead_id} not found",
        context={"lead_id": lead_id},
    )


def _bad_request(msg: str):
    raise_api_error(
        status_code=400,
        code="bad_request",
        message="Bad request",
        detail=msg,
    )


def _normalize_optional_enum(value: str | None, enum_cls, field: str):
    if value is None:
        return None
    try:
        return enum_cls(str(value).strip().upper())
    except Exception:
        allowed = [e.value for e in enum_cls]
        _bad_request(f"Invalid {field}. Allowed values: {allowed}")


# ──────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────

@router.post("", response_model=LeadResponse, status_code=status.HTTP_201_CREATED)
async def create_lead(
    request: Request,
    data: LeadCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    svc: LeadService = Depends(get_lead_service),
):
    """Create a new lead. Source is required; business_domain is optional."""
    if idempotency_key:
        cache_key = f"idem:create_lead:{idempotency_key}"
        cached = await idempotency_store.get(cache_key)
        if cached:
            return cached

    try:
        lead = await svc.create_lead(data)
        if idempotency_key:
            await idempotency_store.set(cache_key, LeadResponse.model_validate(lead).model_dump(mode="json"), ttl_seconds=900)
        return lead
    except DuplicateLeadError as e:
        raise_api_error(
            status_code=409,
            code="duplicate_lead",
            message="Duplicate lead",
            detail=f"A lead with this {e.field} already exists.",
            context={"existing_lead_id": e.existing_id, "field": e.field},
        )


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
    source_normalized = _normalize_optional_enum(source, LeadSource, "source")
    business_domain_normalized = _normalize_optional_enum(business_domain, BusinessDomain, "business_domain")

    offset = (page - 1) * page_size
    items, total = await svc.get_leads(
        stage=stage,
        source=source_normalized,
        business_domain=business_domain_normalized,
        assigned_to_id=assigned_to_id,
        telegram_id=telegram_id,
        offset=offset,
        limit=page_size,
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


@router.get("/cursor", response_model=CursorPageResponse)
async def list_leads_cursor(
    stage: ColdStage | None = Query(default=None),
    source: str | None = Query(default=None, description="Filter by source"),
    business_domain: str | None = Query(default=None, description="Filter by business domain"),
    assigned_to_id: int | None = Query(default=None, description="Filter by assigned user ID"),
    cursor: str | None = Query(default=None, description="Base64-encoded cursor from previous response"),
    limit: int = Query(default=50, ge=1, le=200, description="Items per page"),
    svc: LeadService = Depends(get_lead_service),
):
    """
    List leads with cursor-based pagination.
    
    For large datasets (>10K items), prefer this over page-based pagination.
    Use 'next_cursor' from response to get next page.
    """
    source_normalized = _normalize_optional_enum(source, LeadSource, "source")
    business_domain_normalized = _normalize_optional_enum(business_domain, BusinessDomain, "business_domain")

    # Decode cursor
    cursor_id = None
    if cursor:
        try:
            cursor_data = json.loads(base64.b64decode(cursor).decode())
            cursor_id = cursor_data.get("id")
        except Exception:
            pass  # Invalid cursor, start from beginning
    
    # Get leads using cursor pagination
    items, total, next_cursor = await svc.repo.get_page_by_cursor(
        cursor_id=cursor_id,
        limit=limit,
        stage=stage,
        source=source_normalized,
        business_domain=business_domain_normalized,
        assigned_to_id=assigned_to_id,
    )
    
    # Encode next cursor
    next_cursor_b64 = None
    if next_cursor:
        next_cursor_b64 = base64.b64encode(json.dumps({"id": next_cursor}).encode()).decode()
    
    return {
        "items": items,
        "total": total,
        "next_cursor": next_cursor_b64,
        "has_next": next_cursor is not None,
    }


class BulkActionRequest(BaseModel):
    """Schema for bulk actions on leads."""
    lead_ids: list[int] = Field(..., min_length=1, max_length=100)
    action: str = Field(..., pattern="^(update_stage|delete)$")
    stage: Optional[ColdStage] = None


@router.post("/bulk")
async def bulk_leads_action(
    request: BulkActionRequest,
    svc: LeadService = Depends(get_lead_service),
    role: str = Depends(require_role(UserRole.ADMIN))
):
    """
    Perform bulk actions on multiple leads.
    Step 6.2 — Business Logic
    """
    if request.action == "update_stage":
        if not request.stage:
            raise HTTPException(status_code=400, detail="Stage is required for update_stage action")
        
        count = await svc.repo.bulk_update_stage(request.lead_ids, request.stage)
        await svc.repo.db.commit()
        return {"message": f"Successfully updated {count} leads", "affected": count}
    
    elif request.action == "delete":
        count = await svc.repo.bulk_delete(request.lead_ids)
        await svc.repo.db.commit()
        return {"message": f"Successfully deleted {count} leads", "affected": count}
    
    return {"message": "Invalid action"}


@router.get("/{lead_id}", response_model=LeadResponse)
async def get_lead(
    lead_id: int,
    svc: LeadService = Depends(get_lead_service),
):
    try:
        return await svc.get_lead(lead_id)
    except LeadNotFoundError:
        _not_found(lead_id)


@router.patch("/{lead_id}", response_model=LeadResponse)
async def update_lead(
    lead_id: int,
    data: LeadUpdate,
    svc: LeadService = Depends(get_lead_service),
):
    """Update lead details."""
    try:
        lead = await svc.get_lead(lead_id)
        return await svc.update_lead(lead, data)
    except LeadNotFoundError:
        _not_found(lead_id)


@router.delete("/{lead_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lead(
    lead_id: int,
    svc: LeadService = Depends(get_lead_service),
    _ = Depends(require_role("admin"))
):
    """Soft delete a lead (marks as deleted, preserves history)."""
    from app.core.security import get_current_user
    try:
        lead = await svc.get_lead(lead_id)
        # Get current user for audit
        current_user = get_current_user.__self__ if hasattr(get_current_user, '__self__') else None
        deleted_by = str(current_user) if current_user else "Admin"
        await svc.repo.delete(lead, deleted_by=deleted_by)
        await svc.repo.db.commit()
    except LeadNotFoundError:
        _not_found(lead_id)


@router.post("/{lead_id}/restore", response_model=LeadResponse)
async def restore_lead(
    lead_id: int,
    svc: LeadService = Depends(get_lead_service),
    _ = Depends(require_role("admin"))
):
    """Restore a soft-deleted lead."""
    try:
        lead = await svc.repo.get_by_id(lead_id)
        if not lead:
            _not_found(lead_id)
        if not lead.is_deleted:
            raise HTTPException(status_code=400, detail="Lead is not deleted")
        await svc.repo.restore(lead)
        await svc.repo.db.commit()
        return lead
    except LeadNotFoundError:
        _not_found(lead_id)


@router.get("/deleted", response_model=dict)
async def list_deleted_leads(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    svc: LeadService = Depends(get_lead_service),
    _ = Depends(require_role("admin"))
):
    """List soft-deleted leads (Admin only)."""
    offset = (page - 1) * page_size
    items, total = await svc.repo.get_deleted_leads(offset=offset, limit=page_size)
    total_pages = (total + page_size - 1) // page_size
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


@router.get("/{lead_id}/history", response_model=list[LeadHistoryResponse])
async def get_lead_history(
    lead_id: int,
    history_repo: HistoryRepository = Depends(get_history_repo),
    svc: LeadService = Depends(get_lead_service),
):
    """Get the audit log of stage transitions for a specific lead."""
    try:
        # Check if lead exists first
        await svc.get_lead(lead_id)
        return await history_repo.get_by_lead_id(lead_id)
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
        return await svc.transition_stage(lead, data.stage, lost_reason=data.lost_reason)
    except LeadNotFoundError:
        _not_found(lead_id)
    except LeadStageError as e:
        _bad_request(str(e))
    except MandatoryFieldsError as e:
        _bad_request(
            f"Cannot transition to {e.stage.value}. Missing required fields: {', '.join(e.missing_fields)}. "
            f" Please fill in these fields before advancing the lead."
        )


class StageRollbackRequest(BaseModel):
    """Request body for stage rollback."""
    reason: str = Field(..., min_length=10, description="Reason for rollback (min 10 chars)")


@router.post("/{lead_id}/rollback", response_model=LeadResponse)
async def rollback_stage(
    lead_id: int,
    data: StageRollbackRequest,
    svc: LeadService = Depends(get_lead_service),
    _ = Depends(require_role("manager"))
):
    """
    Rollback lead to previous stage.
    
    Only allowed for CONTACTED → NEW and QUALIFIED → CONTACTED.
    Requires reason with minimum 10 characters.
    """
    try:
        lead = await svc.get_lead(lead_id)
        return await svc.rollback_stage(lead, data.reason)
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
    note_type: str | None = Query(default=None, description="Filter notes by type"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    svc: LeadService = Depends(get_lead_service),
):
    """Get all notes for a lead — SQL-level pagination, never loads all notes into RAM."""
    from sqlalchemy import select, func
    from app.models.note import LeadNote

    # Verify lead exists
    await svc.get_lead(lead_id)
    session = svc.repo.db

    # Count total
    count_query = select(func.count()).select_from(LeadNote).where(LeadNote.lead_id == lead_id)
    if note_type:
        count_query = count_query.where(LeadNote.note_type == note_type)

    count_result = await session.execute(count_query)
    total = count_result.scalar() or 0

    # Paginated fetch
    offset = (page - 1) * page_size
    items_query = (
        select(LeadNote)
        .where(LeadNote.lead_id == lead_id)
        .order_by(LeadNote.is_pinned.desc(), LeadNote.created_at.desc())
        .limit(page_size)
        .offset(offset)
    )
    if note_type:
        items_query = items_query.where(LeadNote.note_type == note_type)

    items_result = await session.execute(items_query)
    page_notes = list(items_result.scalars().all())

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
    request: Request,
    data: NoteCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    svc: LeadService = Depends(get_lead_service),
):
    """Add a note to a lead."""
    from app.models.note import LeadNote

    if idempotency_key:
        cache_key = f"idem:create_note:{lead_id}:{idempotency_key}"
        cached = await idempotency_store.get(cache_key)
        if cached:
            return cached
    
    lead = await svc.get_lead(lead_id)
    
    note = LeadNote(
        lead_id=lead.id,
        content=data.content,
        note_type=data.note_type,
        author_id=data.author_id,
        author_name=data.author_name,
    )
    
    # Save note using existing DI session
    session = svc.repo.db
    session.add(note)
    await session.flush()
    await session.refresh(note)

    if idempotency_key:
        await idempotency_store.set(cache_key, NoteResponse.model_validate(note).model_dump(mode="json"), ttl_seconds=900)
    
    return note


@router.patch("/{lead_id}/notes/{note_id}/pin", response_model=NoteResponse)
async def pin_lead_note(
    lead_id: int,
    note_id: int,
    svc: LeadService = Depends(get_lead_service),
):
    """Pin a note to keep it at the top of lead notes."""
    from sqlalchemy import select
    from app.models.note import LeadNote

    await svc.get_lead(lead_id)
    session = svc.repo.db

    result = await session.execute(
        select(LeadNote).where(LeadNote.id == note_id, LeadNote.lead_id == lead_id)
    )
    note = result.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.is_pinned = True
    await session.flush()
    await session.refresh(note)
    return note


@router.patch("/{lead_id}/notes/{note_id}/unpin", response_model=NoteResponse)
async def unpin_lead_note(
    lead_id: int,
    note_id: int,
    svc: LeadService = Depends(get_lead_service),
):
    """Unpin a previously pinned note."""
    from sqlalchemy import select
    from app.models.note import LeadNote

    await svc.get_lead(lead_id)
    session = svc.repo.db

    result = await session.execute(
        select(LeadNote).where(LeadNote.id == note_id, LeadNote.lead_id == lead_id)
    )
    note = result.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.is_pinned = False
    await session.flush()
    await session.refresh(note)
    return note


@router.delete("/{lead_id}/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lead_note(
    lead_id: int,
    note_id: int,
    svc: LeadService = Depends(get_lead_service),
):
    """Delete a note from a lead."""
    from sqlalchemy import select
    from app.models.note import LeadNote
    
    # Verify lead exists
    await svc.get_lead(lead_id)
    
    session = svc.repo.db
    result = await session.execute(
        select(LeadNote).where(LeadNote.id == note_id, LeadNote.lead_id == lead_id)
    )
    note = result.scalar_one_or_none()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    await session.delete(note)
    await session.flush()
    
    return None


# ──────────────────────────────────────────────
# Lead Assignment
# ──────────────────────────────────────────────

@router.post("/{lead_id}/assign/{user_id}", response_model=LeadResponse)
async def assign_lead(
    lead_id: int,
    user_id: int,
    svc: LeadService = Depends(get_lead_service),
    _ = Depends(require_role("manager"))
):
    """Assign lead to a user/manager."""
    from app.models.user import User
    from sqlalchemy import select, func

    lead = await svc.get_lead(lead_id)
    session = svc.repo.db

    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=400, detail="User is not active")

    # Use real-time COUNT to avoid race condition (Step 1.4)
    from app.models.lead import Lead as LeadModel
    count_res = await session.execute(
        select(func.count()).select_from(LeadModel)
        .where(LeadModel.assigned_to_id == user_id)
        .where(LeadModel.stage.notin_([ColdStage.LOST, ColdStage.TRANSFERRED]))
    )
    live_count = count_res.scalar() or 0

    if live_count >= user.max_leads:
        raise HTTPException(status_code=400, detail=f"User has reached max leads capacity ({user.max_leads})")

    lead.assigned_to_id = user_id
    user.current_leads = live_count + 1   # keep denormalized field in sync

    await session.flush()
    await session.refresh(lead)

    return lead


@router.post("/{lead_id}/unassign", response_model=LeadResponse)
async def unassign_lead(
    lead_id: int,
    svc: LeadService = Depends(get_lead_service),
    _ = Depends(require_role("manager"))
):
    """Unassign lead from user."""
    from app.models.user import User
    from sqlalchemy import select, func
    from app.models.lead import Lead as LeadModel

    lead = await svc.get_lead(lead_id)
    session = svc.repo.db

    if lead.assigned_to_id:
        result = await session.execute(select(User).where(User.id == lead.assigned_to_id))
        user = result.scalar_one_or_none()

        lead.assigned_to_id = None
        await session.flush()  # flush before counting so this lead is excluded

        if user:
            # Recalculate from DB — avoids race condition
            count_res = await session.execute(
                select(func.count()).select_from(LeadModel)
                .where(LeadModel.assigned_to_id == user.id)
                .where(LeadModel.stage.notin_([ColdStage.LOST, ColdStage.TRANSFERRED]))
            )
            user.current_leads = count_res.scalar() or 0

        await session.flush()
        await session.refresh(lead)

    return lead


@router.post("/{lead_id}/reassign/{new_user_id}", response_model=LeadResponse)
async def reassign_lead(
    lead_id: int,
    new_user_id: int,
    svc: LeadService = Depends(get_lead_service),
    _ = Depends(require_role("manager"))
):
    """Reassign lead from current user to another user."""
    from app.models.user import User
    from sqlalchemy import select, func
    from app.models.lead import Lead as LeadModel

    lead = await svc.get_lead(lead_id)
    old_user_id = lead.assigned_to_id
    session = svc.repo.db

    # Get new user
    result = await session.execute(select(User).where(User.id == new_user_id))
    new_user = result.scalar_one_or_none()

    if not new_user:
        raise HTTPException(status_code=404, detail="New user not found")

    if not new_user.is_active:
        raise HTTPException(status_code=400, detail="New user is not active")

    # Real-time capacity check for new user
    new_count_res = await session.execute(
        select(func.count()).select_from(LeadModel)
        .where(LeadModel.assigned_to_id == new_user_id)
        .where(LeadModel.stage.notin_([ColdStage.LOST, ColdStage.TRANSFERRED]))
    )
    new_live_count = new_count_res.scalar() or 0

    if new_live_count >= new_user.max_leads:
        raise HTTPException(status_code=400, detail=f"New user has reached max leads capacity ({new_user.max_leads})")

    # Reassign
    lead.assigned_to_id = new_user_id
    await session.flush()  # flush before recounting

    # Update counts via real-time COUNT to prevent race conditions
    if old_user_id:
        old_result = await session.execute(select(User).where(User.id == old_user_id))
        old_user = old_result.scalar_one_or_none()
        if old_user:
            old_count_res = await session.execute(
                select(func.count()).select_from(LeadModel)
                .where(LeadModel.assigned_to_id == old_user_id)
                .where(LeadModel.stage.notin_([ColdStage.LOST, ColdStage.TRANSFERRED]))
            )
            old_user.current_leads = old_count_res.scalar() or 0

    new_user.current_leads = new_live_count + 1

    await session.flush()
    await session.refresh(lead)

    return lead

# ──────────────────────────────────────────────
# Attachments
# ──────────────────────────────────────────────

ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif",
    "application/pdf", "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/{lead_id}/attachments", response_model=LeadAttachmentResponse)
async def upload_attachment(
    lead_id: int,
    file: UploadFile = File(...),
    svc: LeadService = Depends(get_lead_service),
):
    """Upload a file attachment for a lead (async I/O, 10MB limit, MIME whitelist)."""
    # MIME type check
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"File type '{file.content_type}' is not allowed. Accepted: {sorted(ALLOWED_MIME_TYPES)}",
        )

    try:
        # Verify lead exists
        await svc.get_lead(lead_id)

        # Read content — enforces 10 MB hard limit
        content = await file.read(MAX_FILE_SIZE + 1)
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File exceeds the 10 MB size limit.")

        # Determine file type bucket
        file_ext = os.path.splitext(file.filename or "")[1].lower()
        file_type = "photo" if file_ext in {".jpg", ".jpeg", ".png", ".gif"} else "document"

        # Unique filename — use timezone-aware timestamp (Step 1.2 + 1.6 combined fix)
        safe_name = f"{lead_id}_{int(datetime.now(UTC).timestamp())}_{file.filename}"
        file_path = UPLOAD_DIR / safe_name

        # Async write — does NOT block the event loop
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(content)

        # Create DB record
        attachment = await svc.add_attachment(
            lead_id=lead_id,
            file_name=file.filename,
            file_type=file_type,
            file_path=str(file_path),
            file_size=len(content),
            uploaded_by="BotUser"
        )
        return attachment

    except LeadNotFoundError:
        _not_found(lead_id)
    except HTTPException:
        raise
    except Exception as e:
        _bad_request(str(e))

@router.get("/{lead_id}/attachments", response_model=List[LeadAttachmentResponse])
async def get_attachments(
    lead_id: int,
    svc: LeadService = Depends(get_lead_service)
):
    """List all attachments for a lead."""
    try:
        await svc.get_lead(lead_id)
        return await svc.get_attachments(lead_id)
    except LeadNotFoundError:
        _not_found(lead_id)

