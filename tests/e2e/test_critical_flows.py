"""
E2E Critical Scenario Tests (Step 9)
Tests the critical bot flows as required:
- lead_create
- lead_edit
- lead_note
- lead_save
- lead_transfer

These tests verify that the core CRM functionality works end-to-end.
"""
import pytest
import asyncio
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.base import Base
from app.models.lead import Lead, ColdStage, LeadSource, BusinessDomain
from app.models.sale import Sale, SaleStage
from app.models.user import User, UserRole
from app.models.note import LeadNote
from app.repositories.lead_repo import LeadRepository
from app.repositories.sale_repo import SaleRepository
from app.repositories.user_repo import UserRepository
from app.services.lead_service import LeadService
from app.services.transfer_service import TransferService


# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture
async def db_session():
    """Create a fresh database session for each test."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestingSessionLocal() as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def test_user(db_session):
    """Create a test user/agent."""
    user = User(
        telegram_id="123456789",
        username="test_agent",
        full_name="Test Agent",
        role=UserRole.AGENT,
        is_active=True,
        max_leads=50,
        current_leads=0,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def test_admin(db_session):
    """Create a test admin user."""
    admin = User(
        telegram_id="987654321",
        username="test_admin",
        full_name="Test Admin",
        role=UserRole.ADMIN,
        is_active=True,
        max_leads=100,
        current_leads=0,
    )
    db_session.add(admin)
    await db_session.commit()
    await db_session.refresh(admin)
    return admin


# ──────────────────────────────────────────────
# Critical Scenario: Lead Create
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lead_create(db_session, test_user):
    """Test: lead_create - Create a new lead (Step 9)."""
    lead_repo = LeadRepository(db_session)
    history_repo = MagicMock()  # Mock history repo
    
    lead_service = LeadService(lead_repo, history_repo)
    
    # Create a new lead
    lead_data = {
        "source": LeadSource.MANUAL,
        "full_name": "John Doe",
        "phone": "+380501234567",
        "email": "john@example.com",
        "business_domain": BusinessDomain.FIRST,
    }
    
    new_lead = await lead_service.create_lead(lead_data, created_by="test")
    
    # Verify lead was created
    assert new_lead.id is not None
    assert new_lead.full_name == "John Doe"
    assert new_lead.phone == "+380501234567"
    assert new_lead.email == "john@example.com"
    assert new_lead.source == LeadSource.MANUAL
    assert new_lead.stage == ColdStage.NEW
    assert new_lead.business_domain == BusinessDomain.FIRST
    
    # Verify created timestamps
    assert new_lead.created_at is not None
    assert new_lead.updated_at is not None
    
    print("✓ test_lead_create passed")


# ──────────────────────────────────────────────
# Critical Scenario: Lead Edit
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lead_edit(db_session, test_user):
    """Test: lead_edit - Edit an existing lead (Step 9)."""
    lead_repo = LeadRepository(db_session)
    history_repo = MagicMock()
    
    lead_service = LeadService(lead_repo, history_repo)
    
    # First create a lead
    lead_data = {
        "source": LeadSource.SCANNER,
        "full_name": "Original Name",
        "phone": "+380501234567",
        "business_domain": BusinessDomain.FIRST,
    }
    
    lead = await lead_service.create_lead(lead_data, created_by="test")
    
    # Update the lead
    update_data = {
        "full_name": "Updated Name",
        "business_domain": BusinessDomain.SECOND,
    }
    
    updated_lead = await lead_service.update_lead(lead.id, update_data, updated_by="test")
    
    # Verify lead was updated
    assert updated_lead.full_name == "Updated Name"
    assert updated_lead.business_domain == BusinessDomain.SECOND
    assert updated_lead.source == LeadSource.SCANNER  # Unchanged
    
    print("✓ test_lead_edit passed")


# ──────────────────────────────────────────────
# Critical Scenario: Lead Note
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lead_note(db_session, test_user):
    """Test: lead_note - Add a note to a lead (Step 9)."""
    lead_repo = LeadRepository(db_session)
    history_repo = MagicMock()
    
    lead_service = LeadService(lead_repo, history_repo)
    
    # Create a lead first
    lead_data = {
        "source": LeadSource.PARTNER,
        "full_name": "Jane Doe",
        "phone": "+380509999999",
    }
    
    lead = await lead_service.create_lead(lead_data, created_by="test")
    
    # Add a note to the lead
    note_content = "Customer interested in premium package"
    note = LeadNote(
        lead_id=lead.id,
        content=note_content,
        created_by="test_agent",
    )
    db_session.add(note)
    await db_session.commit()
    await db_session.refresh(note)
    
    # Verify note was added
    assert note.id is not None
    assert note.content == note_content
    assert note.lead_id == lead.id
    
    # Reload lead and verify note is associated
    await db_session.refresh(lead)
    assert len(lead.notes) == 1
    assert lead.notes[0].content == note_content
    
    print("✓ test_lead_note passed")


# ──────────────────────────────────────────────
# Critical Scenario: Lead Save
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lead_save(db_session, test_user):
    """Test: lead_save - Save/preserve lead data (Step 9).
    
    This test verifies that lead data persists correctly in the database.
    """
    lead_repo = LeadRepository(db_session)
    history_repo = MagicMock()
    
    lead_service = LeadService(lead_repo, history_repo)
    
    # Create a lead with all fields
    lead_data = {
        "source": LeadSource.MANUAL,
        "full_name": "Save Test User",
        "phone": "+380501111111",
        "email": "save@test.com",
        "business_domain": BusinessDomain.THIRD,
        "telegram_id": "111222333",
    }
    
    created_lead = await lead_service.create_lead(lead_data, created_by="test")
    
    # Fetch it back from database
    fetched_lead = await lead_service.get_lead(created_lead.id)
    
    # Verify data was saved and can be retrieved
    assert fetched_lead is not None
    assert fetched_lead.id == created_lead.id
    assert fetched_lead.full_name == "Save Test User"
    assert fetched_lead.phone == "+380501111111"
    assert fetched_lead.email == "save@test.com"
    assert fetched_lead.business_domain == BusinessDomain.THIRD
    assert fetched_lead.telegram_id == "111222333"
    assert fetched_lead.source == LeadSource.MANUAL
    
    print("✓ test_lead_save passed")


# ──────────────────────────────────────────────
# Critical Scenario: Lead Transfer
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lead_transfer(db_session, test_user, test_admin):
    """Test: lead_transfer - Transfer lead to sales (Step 9)."""
    lead_repo = LeadRepository(db_session)
    sale_repo = SaleRepository(db_session)
    user_repo = UserRepository(db_session)
    history_repo = MagicMock()
    
    lead_service = LeadService(lead_repo, history_repo)
    transfer_service = TransferService(
        lead_repo=lead_repo,
        sale_repo=sale_repo,
        user_repo=user_repo,
    )
    
    # Create a lead in qualified stage
    lead_data = {
        "source": LeadSource.PARTNER,
        "full_name": "Transfer Test",
        "phone": "+380504444444",
        "business_domain": BusinessDomain.FIRST,
        "stage": ColdStage.QUALIFIED,
    }
    
    lead = await lead_service.create_lead(lead_data, created_by="test")
    
    # Assign to agent
    lead.assigned_to_id = test_user.id
    await lead_repo.save(lead)
    
    # Transfer to sales
    transfer_result = await transfer_service.transfer_to_sales(
        lead_id=lead.id,
        transferred_by=test_admin.full_name,
        notes="High-value client, ready for KYC",
    )
    
    # Verify transfer was successful
    assert transfer_result["success"] is True
    assert "sale_id" in transfer_result
    
    # Verify lead stage changed
    await db_session.refresh(lead)
    assert lead.stage == ColdStage.TRANSFERRED
    
    # Verify sale was created
    sale = await sale_repo.get_by_id(transfer_result["sale_id"])
    assert sale is not None
    assert sale.lead_id == lead.id
    assert sale.stage == SaleStage.NEW
    
    # Verify user stats updated
    await db_session.refresh(test_user)
    assert test_user.sales_converted == 1
    
    print("✓ test_lead_transfer passed")


# ──────────────────────────────────────────────
# Combined E2E Test
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_lead_lifecycle(db_session, test_user, test_admin):
    """Test: Complete lead lifecycle from create to transfer (Step 9).
    
    This test covers the entire flow:
    1. Create lead
    2. Edit lead (add details)
    3. Add note
    4. Save (verify persistence)
    5. Transfer to sales
    """
    lead_repo = LeadRepository(db_session)
    sale_repo = SaleRepository(db_session)
    user_repo = UserRepository(db_session)
    history_repo = MagicMock()
    
    lead_service = LeadService(lead_repo, history_repo)
    transfer_service = TransferService(
        lead_repo=lead_repo,
        sale_repo=sale_repo,
        user_repo=user_repo,
    )
    
    # Step 1: Create
    lead_data = {
        "source": LeadSource.SCANNER,
        "full_name": "Lifecycle Test User",
        "phone": "+380505555555",
    }
    lead = await lead_service.create_lead(lead_data, created_by="test")
    assert lead.id is not None
    print("  ✓ Step 1: Create")
    
    # Step 2: Edit
    updated = await lead_service.update_lead(
        lead.id,
        {"full_name": "Updated Lifecycle User", "business_domain": BusinessDomain.SECOND},
        updated_by="test",
    )
    assert updated.full_name == "Updated Lifecycle User"
    print("  ✓ Step 2: Edit")
    
    # Step 3: Add note
    note = LeadNote(
        lead_id=lead.id,
        content="Initial contact made, interested in services",
        created_by=test_user.full_name,
    )
    db_session.add(note)
    await db_session.commit()
    print("  ✓ Step 3: Note")
    
    # Step 4: Save (verify persistence)
    fetched = await lead_service.get_lead(lead.id)
    assert fetched.full_name == "Updated Lifecycle User"
    assert len(fetched.notes) == 1
    print("  ✓ Step 4: Save")
    
    # Step 5: Transfer to sales
    # First qualify the lead
    await lead_service.update_lead(lead.id, {"stage": ColdStage.QUALIFIED}, updated_by="test")
    lead.assigned_to_id = test_user.id
    await lead_repo.save(lead)
    
    result = await transfer_service.transfer_to_sales(
        lead_id=lead.id,
        transferred_by=test_admin.full_name,
    )
    assert result["success"] is True
    print("  ✓ Step 5: Transfer")
    
    print("✓ test_complete_lead_lifecycle passed")


# ──────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
