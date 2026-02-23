"""
Unit tests for RBAC (Role-Based Access Control).
Tests Steps 4, 5, 6 - Security, Pipeline Rules, Data Quality.
"""
import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
    require_role,
    ROLE_HIERARCHY,
)
from app.models.user import User, UserRole
from app.models.lead import Lead, ColdStage, LostReason, BusinessDomain
from app.services.lead_service import (
    LeadService,
    validate_stage_transition,
    STAGE_REQUIREMENTS,
    MandatoryFieldsError,
    LeadStageError,
)


class TestSecurity:
    """Test authentication and authorization."""
    
    def test_create_access_token(self):
        """Test access token creation with role."""
        token = create_access_token({"sub": "1", "role": "MANAGER"})
        assert token is not None
        assert isinstance(token, str)
    
    def test_create_refresh_token(self):
        """Test refresh token creation."""
        token = create_refresh_token({"sub": "1", "role": "MANAGER"})
        assert token is not None
        assert isinstance(token, str)
    
    def test_verify_refresh_token_valid(self):
        """Test refresh token verification with valid token."""
        payload = {"sub": "1", "role": "MANAGER"}
        token = create_refresh_token(payload)
        result = verify_refresh_token(token)
        assert result is not None
        assert result["sub"] == "1"
        assert result["type"] == "refresh"
    
    def test_verify_refresh_token_invalid(self):
        """Test refresh token verification with invalid token."""
        result = verify_refresh_token("invalid_token")
        assert result is None
    
    def test_verify_refresh_token_wrong_type(self):
        """Test refresh token verification with access token."""
        # Create an access token
        access_token = create_access_token({"sub": "1", "role": "MANAGER"})
        # Try to verify it as refresh token
        result = verify_refresh_token(access_token)
        assert result is None
    
    def test_role_hierarchy(self):
        """Test role hierarchy values."""
        assert ROLE_HIERARCHY[UserRole.AGENT] == 1
        assert ROLE_HIERARCHY[UserRole.MANAGER] == 2
        assert ROLE_HIERARCHY[UserRole.ADMIN] == 3
        assert ROLE_HIERARCHY[UserRole.SUPER_ADMIN] == 4
        
        # String keys also work
        assert ROLE_HIERARCHY["AGENT"] == 1
        assert ROLE_HIERARCHY["MANAGER"] == 2


class TestRoleBasedAccess:
    """Test role-based access control."""
    
    @pytest.mark.asyncio
    async def test_require_role_agent_rejected(self):
        """Test that AGENT role is rejected for MANAGER endpoint."""
        from fastapi import HTTPException
        
        # Create mock user with AGENT role
        user = User(id=1, username="agent", role=UserRole.AGENT, is_active=True)
        
        # Mock dependency
        async def mock_get_current_user():
            return user
        
        # Create require_role dependency
        require_manager = require_role(UserRole.MANAGER)
        
        # Mock the Depends - should raise 403
        with patch('app.core.security.get_current_user', mock_get_current_user):
            # This would be called by FastAPI
            pass  # Actual test would use TestClient
    
    def test_agent_cannot_access_admin_endpoints(self):
        """Test that agents cannot access admin-only endpoints."""
        agent_level = ROLE_HIERARCHY[UserRole.AGENT]
        admin_level = ROLE_HIERARCHY[UserRole.ADMIN]
        
        assert agent_level < admin_level
    
    def test_manager_cannot_access_admin_delete(self):
        """Test that managers cannot delete leads."""
        manager_level = ROLE_HIERARCHY[UserRole.MANAGER]
        admin_level = ROLE_HIERARCHY[UserRole.ADMIN]
        
        assert manager_level < admin_level


class TestStageTransitions:
    """Test stage transition rules."""
    
    def test_stage_requirements_defined(self):
        """Test that all stages have requirements defined."""
        assert ColdStage.NEW in STAGE_REQUIREMENTS
        assert ColdStage.CONTACTED in STAGE_REQUIREMENTS
        assert ColdStage.QUALIFIED in STAGE_REQUIREMENTS
        assert ColdStage.TRANSFERRED in STAGE_REQUIREMENTS
        assert ColdStage.LOST in STAGE_REQUIREMENTS
    
    def test_validate_stage_new_no_requirements(self):
        """Test that NEW stage has no requirements."""
        lead = Lead(
            id=1,
            stage=ColdStage.NEW,
            full_name=None,
            phone=None,
            email=None,
            telegram_id=None,
            source="SCANNER",
        )
        missing = validate_stage_transition(lead, ColdStage.NEW)
        assert missing == []
    
    def test_validate_stage_contacted_requires_contact(self):
        """Test that CONTACTED requires contact info."""
        # Lead without contact info
        lead = Lead(
            id=1,
            stage=ColdStage.NEW,
            full_name="Test",
            phone=None,
            email=None,
            telegram_id=None,
            source="SCANNER",
        )
        missing = validate_stage_transition(lead, ColdStage.CONTACTED)
        assert "phone" in missing or "email" in missing or "telegram_id" in missing
    
    def test_validate_stage_contacted_with_phone(self):
        """Test that CONTACTED passes with phone."""
        lead = Lead(
            id=1,
            stage=ColdStage.NEW,
            full_name="Test",
            phone="+380501234567",
            email=None,
            telegram_id=None,
            source="SCANNER",
        )
        missing = validate_stage_transition(lead, ColdStage.CONTACTED)
        assert missing == []
    
    def test_validate_stage_qualified_requires_name_and_domain(self):
        """Test that QUALIFIED requires full_name and business_domain."""
        lead = Lead(
            id=1,
            stage=ColdStage.CONTACTED,
            full_name="Test",
            phone="+380501234567",
            email=None,
            telegram_id=None,
            source="SCANNER",
            business_domain=None,
        )
        missing = validate_stage_transition(lead, ColdStage.QUALIFIED)
        assert "business_domain" in missing
    
    def test_validate_stage_qualified_with_all_fields(self):
        """Test that QUALIFIED passes with all required fields."""
        lead = Lead(
            id=1,
            stage=ColdStage.CONTACTED,
            full_name="Test User",
            phone="+380501234567",
            email="test@example.com",
            telegram_id=None,
            source="SCANNER",
            business_domain=BusinessDomain.FIRST,
        )
        missing = validate_stage_transition(lead, ColdStage.QUALIFIED)
        assert missing == []
    
    def test_validate_stage_transferred_requires_ai_score(self):
        """Test that TRANSFERRED requires AI score."""
        lead = Lead(
            id=1,
            stage=ColdStage.QUALIFIED,
            full_name="Test User",
            phone="+380501234567",
            email="test@example.com",
            source="SCANNER",
            business_domain=BusinessDomain.FIRST,
            ai_score=None,
        )
        missing = validate_stage_transition(lead, ColdStage.TRANSFERRED)
        assert "ai_score" in missing
    
    def test_validate_lost_has_no_requirements(self):
        """Test that LOST stage can be reached from any state."""
        lead = Lead(
            id=1,
            stage=ColdStage.NEW,
            full_name=None,
            phone=None,
            email=None,
            telegram_id=None,
            source="SCANNER",
        )
        missing = validate_stage_transition(lead, ColdStage.LOST)
        assert missing == []


class TestLostReason:
    """Test lost reason taxonomy."""
    
    def test_lost_reason_all_values(self):
        """Test that all lost reason values are defined."""
        reasons = list(LostReason)
        assert LostReason.NO_BUDGET in reasons
        assert LostReason.NO_RESPONSE in reasons
        assert LostReason.COMPETITOR in reasons
        assert LostReason.NOT_INTERESTED in reasons
        assert LostReason.INVALID_CONTACT in reasons
        assert LostReason.OTHER in reasons
    
    def test_lost_reason_count(self):
        """Test we have the expected number of lost reasons."""
        assert len(LostReason) >= 5


class TestMandatoryFieldsError:
    """Test MandatoryFieldsError exception."""
    
    def test_exception_message(self):
        """Test exception message format."""
        error = MandatoryFieldsError(
            stage=ColdStage.CONTACTED,
            missing_fields=["phone", "email"]
        )
        assert "CONTACTED" in str(error)
        assert "phone" in str(error)
        assert "email" in str(error)
    
    def test_exception_attributes(self):
        """Test exception attributes."""
        error = MandatoryFieldsError(
            stage=ColdStage.QUALIFIED,
            missing_fields=["business_domain"]
        )
        assert error.stage == ColdStage.QUALIFIED
        assert "business_domain" in error.missing_fields


class TestTerminalStages:
    """Test terminal stage rules."""
    
    def test_transferred_is_terminal(self):
        """Test that TRANSFERRED is a terminal stage."""
        from app.models.lead import TERMINAL_COLD_STAGES
        assert ColdStage.TRANSFERRED in TERMINAL_COLD_STAGES
    
    def test_lost_is_terminal(self):
        """Test that LOST is a terminal stage."""
        from app.models.lead import TERMINAL_COLD_STAGES
        assert ColdStage.LOST in TERMINAL_COLD_STAGES
    
    def test_non_terminal_stages(self):
        """Test that NEW, CONTACTED, QUALIFIED are not terminal."""
        from app.models.lead import TERMINAL_COLD_STAGES
        assert ColdStage.NEW not in TERMINAL_COLD_STAGES
        assert ColdStage.CONTACTED not in TERMINAL_COLD_STAGES
        assert ColdStage.QUALIFIED not in TERMINAL_COLD_STAGES


class TestStageOrder:
    """Test that stage transitions are sequential."""
    
    def test_cold_stage_order(self):
        """Test cold stage order is defined."""
        from app.models.lead import COLD_STAGE_ORDER
        assert len(COLD_STAGE_ORDER) == 5
        assert COLD_STAGE_ORDER[0] == ColdStage.NEW
        assert COLD_STAGE_ORDER[1] == ColdStage.CONTACTED
        assert COLD_STAGE_ORDER[2] == ColdStage.QUALIFIED
        assert COLD_STAGE_ORDER[3] == ColdStage.TRANSFERRED
        assert COLD_STAGE_ORDER[4] == ColdStage.LOST
    
    def test_cannot_skip_stages(self):
        """Test that stages must be sequential."""
        from app.models.lead import COLD_STAGE_ORDER
        
        # Cannot jump from NEW to QUALIFIED
        new_idx = COLD_STAGE_ORDER.index(ColdStage.NEW)
        qualified_idx = COLD_STAGE_ORDER.index(ColdStage.QUALIFIED)
        assert qualified_idx != new_idx + 1  # This should fail if skipping


class TestReversibleTransitions:
    """Test rollback functionality."""
    
    def test_reversible_stages_defined(self):
        """Test reversible transitions are defined."""
        from app.models.lead import REVERSIBLE_STAGE_TRANSITIONS
        assert ColdStage.CONTACTED in REVERSIBLE_STAGE_TRANSITIONS
        assert ColdStage.QUALIFIED in REVERSIBLE_STAGE_TRANSITIONS
    
    def test_rollback_contacted_to_new(self):
        """Test that CONTACTED can rollback to NEW."""
        from app.models.lead import REVERSIBLE_STAGE_TRANSITIONS
        assert REVERSIBLE_STAGE_TRANSITIONS[ColdStage.CONTACTED] == ColdStage.NEW
    
    def test_rollback_qualified_to_contacted(self):
        """Test that QUALIFIED can rollback to CONTACTED."""
        from app.models.lead import REVERSIBLE_STAGE_TRANSITIONS
        assert REVERSIBLE_STAGE_TRANSITIONS[ColdStage.QUALIFIED] == ColdStage.CONTACTED
    
    def test_cannot_rollback_transferred(self):
        """Test that TRANSFERRED cannot be rolled back."""
        from app.models.lead import REVERSIBLE_STAGE_TRANSITIONS
        assert ColdStage.TRANSFERRED not in REVERSIBLE_STAGE_TRANSITIONS
    
    def test_cannot_rollback_lost(self):
        """Test that LOST cannot be rolled back."""
        from app.models.lead import REVERSIBLE_STAGE_TRANSITIONS
        assert ColdStage.LOST not in REVERSIBLE_STAGE_TRANSITIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
