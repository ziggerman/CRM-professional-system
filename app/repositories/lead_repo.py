"""
Lead Repository - Data Access Layer for Lead model.
"""
from datetime import datetime, UTC
from typing import Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.lead import Lead, ColdStage, LeadSource, BusinessDomain


class LeadRepository:
    """Repository for Lead CRUD operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(self, lead: Lead) -> Lead:
        """Create a new lead."""
        self.db.add(lead)
        await self.db.flush()
        await self.db.refresh(lead)
        return lead
    
    async def get_by_id(self, lead_id: int, include_deleted: bool = False) -> Optional[Lead]:
        """Get lead by ID."""
        stmt = select(Lead).options(selectinload(Lead.sale)).where(Lead.id == lead_id)
        
        # Soft delete filter - by default don't return deleted leads
        if not include_deleted:
            stmt = stmt.where(Lead.is_deleted == False)
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        stage: Optional[ColdStage] = None,
        source: Optional[str] = None,
        business_domain: Optional[str] = None,
        assigned_to_id: Optional[int] = None,
        telegram_id: Optional[str] = None,
        query: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
        include_deleted: bool = False,
    ) -> tuple[list[Lead], int]:
        """Get leads with optional filtering and pagination."""
        from sqlalchemy import String, or_, func
        
        # Base statement for both count and results
        stmt = select(Lead)

        # Soft delete filter - by default exclude deleted leads
        if not include_deleted:
            stmt = stmt.where(Lead.is_deleted == False)

        if stage:
            stmt = stmt.where(Lead.stage == stage)
        if source:
            stmt = stmt.where(Lead.source == source)
        if business_domain:
            stmt = stmt.where(Lead.business_domain == business_domain)
        if assigned_to_id is not None:
            stmt = stmt.where(Lead.assigned_to_id == assigned_to_id)
        if telegram_id:
            stmt = stmt.where(Lead.telegram_id == telegram_id)
        
        if created_after:
            stmt = stmt.where(Lead.created_at >= created_after)
        if created_before:
            stmt = stmt.where(Lead.created_at <= created_before)
            
        if query:
            q_pattern = f"%{query}%"
            stmt = stmt.where(
                or_(
                    Lead.telegram_id.ilike(q_pattern),
                    Lead.source.cast(String).ilike(q_pattern),
                    Lead.business_domain.cast(String).ilike(q_pattern)
                )
            )

        # Total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar() or 0

        # Paginated results with relationships
        stmt = stmt.options(
            selectinload(Lead.sale), 
            selectinload(Lead.notes),
            selectinload(Lead.attachments)
        )
        stmt = stmt.offset(offset).limit(limit).order_by(Lead.created_at.desc())
        
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())

        return leads, total
    
    async def save(self, lead: Lead) -> Lead:
        """Save lead changes."""
        await self.db.flush()
        await self.db.refresh(lead)
        return lead
    
    async def delete(self, lead: Lead, deleted_by: str = "System") -> None:
        """Soft delete a lead - marks as deleted without removing from DB."""
        lead.is_deleted = True
        lead.deleted_at = datetime.now(UTC)
        lead.deleted_by = deleted_by
        await self.db.flush()

    async def restore(self, lead: Lead) -> None:
        """Restore a soft-deleted lead."""
        lead.is_deleted = False
        lead.deleted_at = None
        lead.deleted_by = None
        await self.db.flush()

    async def get_deleted_leads(self, offset: int = 0, limit: int = 50) -> tuple[list[Lead], int]:
        """Get soft-deleted leads (for admin restore UI)."""
        from sqlalchemy import func
        
        stmt = select(Lead).where(Lead.is_deleted == True)
        
        # Total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar() or 0

        # Paginated results
        stmt = stmt.options(
            selectinload(Lead.sale), 
            selectinload(Lead.notes),
            selectinload(Lead.attachments)
        )
        stmt = stmt.offset(offset).limit(limit).order_by(Lead.deleted_at.desc())
        
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())

        return leads, total

    async def get_stale_leads(self, days: int = 7) -> list[Lead]:
        """Get leads in NEW or CONTACTED stages with no updates for N days."""
        from datetime import datetime, UTC, timedelta
        cutoff = datetime.now(UTC) - timedelta(days=days)
        
        stmt = (
            select(Lead)
            .where(Lead.is_deleted == False)  # Exclude deleted leads
            .where(Lead.stage.in_([ColdStage.NEW, ColdStage.CONTACTED]))
            .where(Lead.updated_at <= cutoff)
            .order_by(Lead.updated_at.asc())
        )
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def archive_old_lost_leads(self, days: int = 90) -> int:
        """
        Delete leads that have been in 'lost' stage for more than N days.
        Returns number of archived leads.
        """
        from datetime import datetime, UTC, timedelta
        from sqlalchemy import delete
        cutoff = datetime.now(UTC) - timedelta(days=days)
        
        stmt = (
            delete(Lead)
            .where(Lead.stage == ColdStage.LOST)
            .where(Lead.updated_at <= cutoff)
        )
        
        result = await self.db.execute(stmt)
        return result.rowcount

    async def bulk_update_stage(self, lead_ids: list[int], stage: ColdStage) -> int:
        """Bulk update stage for multiple leads (Step 6.2)."""
        stmt = select(Lead).where(Lead.id.in_(lead_ids))
        result = await self.db.execute(stmt)
        leads = result.scalars().all()
        
        for lead in leads:
            lead.stage = stage
            lead.updated_at = datetime.now(UTC)
        
        return len(leads)

    async def bulk_delete(self, lead_ids: list[int]) -> int:
        """Bulk delete multiple leads (Step 6.2)."""
        from sqlalchemy import delete
        stmt = delete(Lead).where(Lead.id.in_(lead_ids))
        result = await self.db.execute(stmt)
        return result.rowcount

    # ──────────────────────────────────────────────
    # Cursor-based pagination (Phase 5.3)
    # ──────────────────────────────────────────────
    
    async def get_page_by_cursor(
        self,
        cursor_id: Optional[int] = None,
        limit: int = 50,
        stage: Optional[ColdStage] = None,
        source: Optional[str] = None,
        business_domain: Optional[str] = None,
        assigned_to_id: Optional[int] = None,
        include_deleted: bool = False,
    ) -> tuple[list[Lead], int, Optional[str]]:
        """
        Get leads using cursor-based pagination for better performance with large datasets.
        
        Args:
            cursor_id: Last lead ID from previous page (keyset pagination)
            limit: Number of items per page
            stage: Filter by stage
            source: Filter by source
            business_domain: Filter by business domain
            assigned_to_id: Filter by assigned user
            include_deleted: Include soft-deleted leads
            
        Returns:
            Tuple of (leads, total_count, next_cursor)
        """
        from sqlalchemy import String, or_, func
        
        # Base query with filters
        stmt = select(Lead)
        
        # Soft delete filter
        if not include_deleted:
            stmt = stmt.where(Lead.is_deleted == False)
        
        # Apply filters
        if stage:
            stmt = stmt.where(Lead.stage == stage)
        if source:
            stmt = stmt.where(Lead.source == source)
        if business_domain:
            stmt = stmt.where(Lead.business_domain == business_domain)
        if assigned_to_id is not None:
            stmt = stmt.where(Lead.assigned_to_id == assigned_to_id)
        
        # Cursor-based filtering (keyset pagination on ID)
        if cursor_id is not None:
            stmt = stmt.where(Lead.id < cursor_id)
        
        # Total count (without cursor filter for accuracy)
        count_stmt = select(func.count()).select_from(stmt.subquery())
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar() or 0
        
        # Fetch one extra to check if there's a next page
        stmt = stmt.options(
            selectinload(Lead.sale), 
            selectinload(Lead.notes),
            selectinload(Lead.attachments)
        )
        stmt = stmt.order_by(Lead.id.desc()).limit(limit + 1)
        
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())
        
        # Determine next cursor
        next_cursor = None
        has_next = False
        if len(leads) > limit:
            leads = leads[:limit]  # Remove the extra item
            has_next = True
            next_cursor = leads[-1].id  # Last item's ID
        
        return leads, total, next_cursor

    # ──────────────────────────────────────────────
    # Step 6: Lead Deduplication
    # ──────────────────────────────────────────────
    
    async def find_duplicates(
        self,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        telegram_id: Optional[str] = None,
    ) -> list[Lead]:
        """
        Find potential duplicate leads by contact fields.
        Used for soft deduplication - warns about potential duplicates.
        """
        if not any([phone, email, telegram_id]):
            return []
        
        conditions = []
        if phone:
            conditions.append(Lead.phone == phone)
        if email:
            conditions.append(Lead.email == email)
        if telegram_id:
            conditions.append(Lead.telegram_id == telegram_id)
        
        stmt = (
            select(Lead)
            .where(Lead.is_deleted == False)
            .where(or_(*conditions))
            .order_by(Lead.created_at.desc())
        )
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def check_hard_duplicate(
        self,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        telegram_id: Optional[str] = None,
    ) -> Optional[Lead]:
        """
        Check for exact duplicate using all available fields.
        Used for hard deduplication - blocks creation of duplicates.
        Returns the existing lead if found, None otherwise.
        """
        if not any([phone, email, telegram_id]):
            return None
        
        # For hard duplicate, we require exact match on at least 2 fields
        # or exact match on one field plus non-null
        match_count = 0
        if phone:
            match_count += 1
        if email:
            match_count += 1
        if telegram_id:
            match_count += 1
        
        # Need at least 2 fields to match for hard duplicate
        if match_count < 2:
            return None
        
        conditions = []
        if phone:
            conditions.append(Lead.phone == phone)
        if email:
            conditions.append(Lead.email == email)
        if telegram_id:
            conditions.append(Lead.telegram_id == telegram_id)
        
        stmt = (
            select(Lead)
            .where(Lead.is_deleted == False)
            .where(or_(*conditions))
            .order_by(Lead.created_at.desc())
            .limit(1)
        )
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def merge_duplicates(self, primary_id: int, duplicate_ids: list[int]) -> Lead:
        """
        Merge duplicate leads into a primary lead.
        - Moves notes, history, attachments from duplicates to primary
        - Marks duplicates as deleted
        - Returns the updated primary lead
        """
        primary = await self.get_by_id(primary_id)
        if not primary:
            raise ValueError(f"Primary lead {primary_id} not found")
        
        from app.models.note import LeadNote
        from app.models.history import LeadHistory
        from app.models.attachment import LeadAttachment
        
        for dup_id in duplicate_ids:
            dup = await self.get_by_id(dup_id)
            if not dup:
                continue
            
            # Merge notes
            notes_stmt = select(LeadNote).where(LeadNote.lead_id == dup_id)
            notes_result = await self.db.execute(notes_stmt)
            notes = notes_result.scalars().all()
            for note in notes:
                note.lead_id = primary_id
            
            # Merge attachments
            attach_stmt = select(LeadAttachment).where(LeadAttachment.lead_id == dup_id)
            attach_result = await self.db.execute(attach_stmt)
            attachments = attach_result.scalars().all()
            for att in attachments:
                att.lead_id = primary_id
            
            # Mark as merged (soft delete with reason)
            dup.is_deleted = True
            dup.deleted_by = f"merged_into_{primary_id}"
            dup.deleted_at = datetime.now(UTC)
        
        await self.db.flush()
        await self.db.refresh(primary)
        return primary
