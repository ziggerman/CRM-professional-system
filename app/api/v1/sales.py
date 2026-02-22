"""
Sales API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from starlette import status

from app.core.deps import get_sale_repo, get_transfer_service
from app.models.sale import SaleStage
from app.schemas.sale import (
    SaleResponse,
    SaleListResponse,
    SaleStageUpdate,
    SaleUpdate,
)
from app.repositories.sale_repo import SaleRepository
from app.services.transfer_service import TransferService, TransferError

from app.core.security import get_current_user, require_role
from app.models.user import User


router = APIRouter(dependencies=[Depends(get_current_user)])


def _not_found(sale_id: int):
    raise HTTPException(status_code=404, detail=f"Sale {sale_id} not found")


def _bad_request(msg: str):
    raise HTTPException(status_code=400, detail=msg)


# ──────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────

@router.get("", response_model=SaleListResponse)
async def list_sales(
    stage: SaleStage | None = Query(default=None),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page"),
    repo: SaleRepository = Depends(get_sale_repo),
):
    """List sales with pagination."""
    offset = (page - 1) * page_size
    items, total = await repo.get_all(stage=stage, offset=offset, limit=page_size)
    
    total_pages = (total + page_size - 1) // page_size
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


@router.get("/{sale_id}", response_model=SaleResponse)
async def get_sale(
    sale_id: int,
    repo: SaleRepository = Depends(get_sale_repo),
):
    """Get a specific sale by ID."""
    sale = await repo.get_by_id(sale_id)
    if not sale:
        _not_found(sale_id)
    return sale


# ──────────────────────────────────────────────
# Stage management
# ──────────────────────────────────────────────

@router.patch("/{sale_id}/stage", response_model=SaleResponse)
async def update_sale_stage(
    sale_id: int,
    data: SaleStageUpdate,
    repo: SaleRepository = Depends(get_sale_repo),
    transfer_svc: TransferService = Depends(get_transfer_service),
):
    """
    Advance sale to next stage.
    Rules: sequential only, terminal stages locked.
    """
    sale = await repo.get_by_id(sale_id)
    if not sale:
        _not_found(sale_id)
    
    try:
        return await transfer_svc.advance_sale_stage(sale, data.stage)
    except TransferError as e:
        _bad_request(str(e))


@router.patch("/{sale_id}", response_model=SaleResponse)
async def update_sale(
    sale_id: int,
    data: SaleUpdate,
    repo: SaleRepository = Depends(get_sale_repo),
):
    """
    Update sale details (amount, notes).
    """
    sale = await repo.get_by_id(sale_id)
    if not sale:
        _not_found(sale_id)
    
    if data.amount is not None:
        sale.amount = data.amount
    if data.notes is not None:
        sale.notes = data.notes
    
    return await repo.save(sale)


# ──────────────────────────────────────────────
# Lead relationship
# ──────────────────────────────────────────────

@router.get("/lead/{lead_id}", response_model=SaleResponse)
async def get_sale_by_lead(
    lead_id: int,
    repo: SaleRepository = Depends(get_sale_repo),
):
    """
    Get sale associated with a specific lead.
    Useful for checking if a lead has been transferred.
    """
    sale = await repo.get_by_lead_id(lead_id)
    if not sale:
        raise HTTPException(
            status_code=404, 
            detail=f"No sale found for lead {lead_id}"
        )
    return sale
