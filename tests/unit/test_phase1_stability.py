"""
Unit Tests — Phase 1 Stability Verification
Tests cover all 7 bug fixes from ACTION_PLAN.md Phase 1.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC


# ──────────────────────────────────────────────
# Step 1.1 — AI prompts constants must match model enums
# ──────────────────────────────────────────────

class TestAIPromptsSync:
    """Verify that prompts.py constants are always in sync with model enums."""

    def test_all_lead_sources_are_valid_for_ai(self):
        """Every LeadSource enum value must pass _validate_lead_features without raising."""
        from app.models.lead import LeadSource, ColdStage
        from app.ai.prompts import _validate_lead_features

        for source in LeadSource:
            features = {
                "source": source.value,
                "stage": ColdStage.NEW.value,
                "message_count": 0,
                "days_since_created": 1,
            }
            # Must NOT raise ValueError
            _validate_lead_features(features)

    def test_all_cold_stages_are_valid_for_ai(self):
        """Every ColdStage enum value must pass _validate_lead_features without raising."""
        from app.models.lead import LeadSource, ColdStage
        from app.ai.prompts import _validate_lead_features

        for stage in ColdStage:
            features = {
                "source": LeadSource.MANUAL.value,
                "stage": stage.value,
                "message_count": 0,
                "days_since_created": 1,
            }
            _validate_lead_features(features)

    def test_valid_lead_sources_contains_scanner(self):
        from app.ai.prompts import VALID_LEAD_SOURCES
        assert "SCANNER" in VALID_LEAD_SOURCES

    def test_valid_lead_sources_contains_partner(self):
        from app.ai.prompts import VALID_LEAD_SOURCES
        assert "PARTNER" in VALID_LEAD_SOURCES

    def test_valid_lead_stages_does_not_contain_negotiation(self):
        """Old stale constant NEGOTIATION must be gone."""
        from app.ai.prompts import VALID_LEAD_STAGES
        assert "NEGOTIATION" not in VALID_LEAD_STAGES

    def test_valid_lead_stages_does_not_contain_closed(self):
        """Old stale constant CLOSED must be gone."""
        from app.ai.prompts import VALID_LEAD_STAGES
        assert "CLOSED" not in VALID_LEAD_STAGES

    def test_valid_lead_stages_matches_cold_stage_enum(self):
        """VALID_LEAD_STAGES must be exactly the set of ColdStage values."""
        from app.models.lead import ColdStage
        from app.ai.prompts import VALID_LEAD_STAGES
        expected = frozenset(e.value for e in ColdStage)
        assert VALID_LEAD_STAGES == expected


# ──────────────────────────────────────────────
# Step 1.6 — Transfer requires QUALIFIED stage
# ──────────────────────────────────────────────

class TestTransferValidation:
    """Transfer gate must enforce QUALIFIED stage."""

    def _make_lead(self, stage, ai_score=0.75, business_domain="FIRST"):
        from app.models.lead import Lead, ColdStage, LeadSource, BusinessDomain
        lead = MagicMock(spec=Lead)
        lead.id = 1
        lead.stage = stage
        lead.ai_score = ai_score
        lead.ai_recommendation = "transfer_to_sales"
        lead.ai_reason = "mock"
        lead.ai_analyzed_at = datetime.now(UTC)
        lead.business_domain = business_domain
        return lead

    @pytest.mark.asyncio
    async def test_transfer_from_contacted_raises(self):
        from app.models.lead import ColdStage
        from app.services.transfer_service import TransferService, TransferError

        lead = self._make_lead(ColdStage.CONTACTED)
        svc = TransferService.__new__(TransferService)

        with pytest.raises(TransferError, match="QUALIFIED"):
            await svc.transfer_to_sales(lead)

    @pytest.mark.asyncio
    async def test_transfer_from_new_raises(self):
        from app.models.lead import ColdStage
        from app.services.transfer_service import TransferService, TransferError

        lead = self._make_lead(ColdStage.NEW)
        svc = TransferService.__new__(TransferService)

        with pytest.raises(TransferError, match="QUALIFIED"):
            await svc.transfer_to_sales(lead)

    @pytest.mark.asyncio
    async def test_transfer_already_transferred_raises(self):
        from app.models.lead import ColdStage
        from app.services.transfer_service import TransferService, TransferError

        lead = self._make_lead(ColdStage.TRANSFERRED)
        svc = TransferService.__new__(TransferService)

        with pytest.raises(TransferError, match="already transferred"):
            await svc.transfer_to_sales(lead)

    @pytest.mark.asyncio
    async def test_transfer_low_score_raises(self):
        from app.models.lead import ColdStage
        from app.services.transfer_service import TransferService, TransferError

        lead = self._make_lead(ColdStage.QUALIFIED, ai_score=0.3)
        svc = TransferService.__new__(TransferService)

        with pytest.raises(TransferError, match="below minimum"):
            await svc.transfer_to_sales(lead)

    @pytest.mark.asyncio
    async def test_transfer_missing_domain_raises(self):
        from app.models.lead import ColdStage
        from app.services.transfer_service import TransferService, TransferError

        lead = self._make_lead(ColdStage.QUALIFIED, business_domain=None)
        svc = TransferService.__new__(TransferService)

        with pytest.raises(TransferError, match="business_domain"):
            await svc.transfer_to_sales(lead)


# ──────────────────────────────────────────────
# Stage Machine invariants
# ──────────────────────────────────────────────

class TestStageMachine:
    """Stage transitions must follow the sequential ordering rules."""

    def _make_service(self):
        from app.services.lead_service import LeadService
        repo = MagicMock()
        repo.db = MagicMock()
        repo.db.add = MagicMock()
        repo.save = AsyncMock(side_effect=lambda x: x)
        history_repo = MagicMock()
        return LeadService(repo, history_repo)

    def _make_lead(self, stage):
        from app.models.lead import Lead, LeadSource, ColdStage
        lead = MagicMock(spec=Lead)
        lead.id = 1
        lead.stage = stage
        return lead

    @pytest.mark.asyncio
    async def test_new_to_contacted_ok(self):
        from app.models.lead import ColdStage
        svc = self._make_service()
        lead = self._make_lead(ColdStage.NEW)

        result = await svc.transition_stage(lead, ColdStage.CONTACTED)
        assert lead.stage == ColdStage.CONTACTED

    @pytest.mark.asyncio
    async def test_skip_stage_raises(self):
        from app.models.lead import ColdStage
        from app.services.lead_service import LeadStageError
        svc = self._make_service()
        lead = self._make_lead(ColdStage.NEW)

        with pytest.raises(LeadStageError, match="Expected next stage"):
            await svc.transition_stage(lead, ColdStage.QUALIFIED)

    @pytest.mark.asyncio
    async def test_terminal_transferred_locked(self):
        from app.models.lead import ColdStage
        from app.services.lead_service import LeadStageError
        svc = self._make_service()
        lead = self._make_lead(ColdStage.TRANSFERRED)

        with pytest.raises(LeadStageError, match="terminal stage"):
            await svc.transition_stage(lead, ColdStage.LOST)

    @pytest.mark.asyncio
    async def test_terminal_lost_locked(self):
        from app.models.lead import ColdStage
        from app.services.lead_service import LeadStageError
        svc = self._make_service()
        lead = self._make_lead(ColdStage.LOST)

        with pytest.raises(LeadStageError, match="terminal stage"):
            await svc.transition_stage(lead, ColdStage.NEW)

    @pytest.mark.asyncio
    async def test_any_nonterminal_to_lost_ok(self):
        from app.models.lead import ColdStage
        svc = self._make_service()

        for stage in [ColdStage.NEW, ColdStage.CONTACTED, ColdStage.QUALIFIED]:
            lead = self._make_lead(stage)
            await svc.transition_stage(lead, ColdStage.LOST)
            assert lead.stage == ColdStage.LOST


# ──────────────────────────────────────────────
# Step 1.7 — Duplicate lead detection
# ──────────────────────────────────────────────

class TestDuplicateLeadDetection:
    """create_lead must raise DuplicateLeadError if email or phone already exists."""

    @pytest.mark.asyncio
    async def test_duplicate_email_raises(self):
        from app.services.lead_service import LeadService, DuplicateLeadError
        from app.schemas.lead import LeadCreate
        from app.models.lead import LeadSource

        # Mock DB returning existing row
        existing_row = MagicMock()
        existing_row.id = 42
        existing_row.email = "test@example.com"
        existing_row.phone = None

        mock_result = MagicMock()
        mock_result.first = MagicMock(return_value=existing_row)

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = MagicMock()
        repo.db = mock_session

        svc = LeadService(repo, MagicMock())

        data = MagicMock(spec=LeadCreate)
        data.email = "test@example.com"
        data.phone = None
        data.source = LeadSource.MANUAL
        data.business_domain = None
        data.telegram_id = None
        data.full_name = None
        data.external_username = None
        data.intent = None
        data.company = None
        data.position = None
        data.budget = None
        data.pain_points = None

        with pytest.raises(DuplicateLeadError) as exc_info:
            await svc.create_lead(data)

        assert exc_info.value.existing_id == 42
        assert exc_info.value.field == "email"
