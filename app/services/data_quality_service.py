"""
Data Quality Service - Step 6: Quality checks for leads
Provides utilities to identify and flag leads with data quality issues.
"""
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.lead import Lead, ColdStage
from app.models.history import LeadHistory


class DataQualityIssue(str, Enum):
    """Types of data quality issues."""
    MISSING_CONTACT = "missing_contact"  # No phone/email/telegram
    MISSING_NAME = "missing_name"        # No full_name
    MISSING_DOMAIN = "missing_domain"    # No business_domain
    INVALID_EMAIL = "invalid_email"
    INVALID_PHONE = "invalid_phone"
    STALE_LEAD = "stale_lead"            # No updates for N days
    NO_ACTIVITY = "no_activity"          # message_count = 0
    LOW_AI_SCORE = "low_ai_score"        # Score below threshold


@dataclass
class LeadQualityIssue:
    """Represents a data quality issue with a lead."""
    lead_id: int
    issue: DataQualityIssue
    severity: str  # "high", "medium", "low"
    description: str


@dataclass
class DataQualityReport:
    """Summary report of data quality across all leads."""
    total_leads: int
    leads_with_issues: int
    issues_by_type: Dict[DataQualityIssue, int]
    stale_leads_count: int
    leads_without_contact: int
    duplicate_warnings: int
    quality_score: float  # 0-100


class DataQualityService:
    """
    Service for analyzing and reporting on lead data quality.
    Step 6 — Quality checks for dashboard.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def check_lead_quality(self, lead: Lead) -> List[LeadQualityIssue]:
        """
        Check a single lead for data quality issues.
        Returns list of issues found.
        """
        issues = []
        
        # Check for contact info
        has_contact = any([
            lead.phone and lead.phone.strip(),
            lead.email and lead.email.strip(),
            lead.telegram_id and lead.telegram_id.strip()
        ])
        
        if not has_contact:
            issues.append(LeadQualityIssue(
                lead_id=lead.id,
                issue=DataQualityIssue.MISSING_CONTACT,
                severity="high",
                description="Lead has no contact information (phone, email, or telegram)"
            ))
        
        # Check for name
        if not lead.full_name or not lead.full_name.strip():
            issues.append(LeadQualityIssue(
                lead_id=lead.id,
                issue=DataQualityIssue.MISSING_NAME,
                severity="high",
                description="Lead has no full name"
            ))
        
        # Check for business domain (required for QUALIFIED stage)
        if lead.stage in [ColdStage.QUALIFIED, ColdStage.TRANSFERRED]:
            if not lead.business_domain:
                issues.append(LeadQualityIssue(
                    lead_id=lead.id,
                    issue=DataQualityIssue.MISSING_DOMAIN,
                    severity="high",
                    description=f"Lead in {lead.stage.value} stage but missing business_domain"
                ))
        
        # Validate email format
        if lead.email:
            import re
            email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
            if not re.match(email_pattern, lead.email):
                issues.append(LeadQualityIssue(
                    lead_id=lead.id,
                    issue=DataQualityIssue.INVALID_EMAIL,
                    severity="medium",
                    description=f"Email format appears invalid: {lead.email}"
                ))
        
        # Validate phone format
        if lead.phone:
            import re
            phone_pattern = r'^\+?[\d\s\-\(\)]{7,20}$'
            if not re.match(phone_pattern, lead.phone):
                issues.append(LeadQualityIssue(
                    lead_id=lead.id,
                    issue=DataQualityIssue.INVALID_PHONE,
                    severity="medium",
                    description=f"Phone format appears invalid: {lead.phone}"
                ))
        
        # Check for stale leads (no updates in N days)
        stale_days = 14
        cutoff = datetime.now(UTC) - timedelta(days=stale_days)
        if lead.updated_at and lead.updated_at < cutoff:
            issues.append(LeadQualityIssue(
                lead_id=lead.id,
                issue=DataQualityIssue.STALE_LEAD,
                severity="medium",
                description=f"No updates for {(datetime.now(UTC) - lead.updated_at).days} days"
            ))
        
        # Check for no activity
        if lead.message_count == 0:
            issues.append(LeadQualityIssue(
                lead_id=lead.id,
                issue=DataQualityIssue.NO_ACTIVITY,
                severity="low",
                description="Lead has no recorded communications"
            ))
        
        # Check for low AI score (if analyzed)
        if lead.ai_score is not None and lead.ai_score < 0.3:
            issues.append(LeadQualityIssue(
                lead_id=lead.id,
                issue=DataQualityIssue.LOW_AI_SCORE,
                severity="low",
                description=f"AI score is low ({lead.ai_score:.2f})"
            ))
        
        return issues
    
    async def get_quality_report(self, days: int = 30) -> DataQualityReport:
        """
        Generate a comprehensive data quality report.
        """
        # Get total leads
        total_stmt = select(func.count()).select_from(Lead).where(Lead.is_deleted == False)
        total_result = await self.db.execute(total_stmt)
        total_leads = total_result.scalar() or 0
        
        # Get leads with no contact info
        no_contact_stmt = select(func.count()).select_from(Lead).where(
            and_(
                Lead.is_deleted == False,
                or_(
                    Lead.phone.is_(None),
                    Lead.email.is_(None),
                    Lead.telegram_id.is_(None)
                ),
                or_(
                    Lead.phone == "",
                    Lead.email == "",
                    Lead.telegram_id == ""
                )
            )
        )
        no_contact_result = await self.db.execute(no_contact_stmt)
        leads_without_contact = no_contact_result.scalar() or 0
        
        # Get stale leads
        stale_cutoff = datetime.now(UTC) - timedelta(days=days)
        stale_stmt = select(func.count()).select_from(Lead).where(
            and_(
                Lead.is_deleted == False,
                Lead.updated_at < stale_cutoff,
                Lead.stage.in_([ColdStage.NEW, ColdStage.CONTACTED])
            )
        )
        stale_result = await self.db.execute(stale_stmt)
        stale_leads_count = stale_result.scalar() or 0
        
        # Get leads with issues (missing critical data)
        issues_by_type = {
            DataQualityIssue.MISSING_CONTACT: 0,
            DataQualityIssue.MISSING_NAME: 0,
            DataQualityIssue.MISSING_DOMAIN: 0,
            DataQualityIssue.INVALID_EMAIL: 0,
            DataQualityIssue.INVALID_PHONE: 0,
            DataQualityIssue.STALE_LEAD: 0,
            DataQualityIssue.NO_ACTIVITY: 0,
            DataQualityIssue.LOW_AI_SCORE: 0,
        }
        
        # Sample leads for detailed analysis
        sample_stmt = select(Lead).where(Lead.is_deleted == False).limit(1000)
        sample_result = await self.db.execute(sample_stmt)
        leads = list(sample_result.scalars().all())
        
        leads_with_issues = 0
        for lead in leads:
            issues = await self.check_lead_quality(lead)
            if issues:
                leads_with_issues += 1
                for issue in issues:
                    if issue.issue in issues_by_type:
                        issues_by_type[issue.issue] += 1
        
        # Calculate quality score (simple metric)
        if total_leads > 0:
            quality_score = ((total_leads - leads_with_issues) / total_leads) * 100
        else:
            quality_score = 100.0
        
        return DataQualityReport(
            total_leads=total_leads,
            leads_with_issues=leads_with_issues,
            issues_by_type=issues_by_type,
            stale_leads_count=stale_leads_count,
            leads_without_contact=leads_without_contact,
            duplicate_warnings=0,  # Would require separate deduplication check
            quality_score=quality_score
        )
    
    async def get_leads_by_quality_issue(
        self,
        issue: DataQualityIssue,
        limit: int = 50,
        offset: int = 0
    ) -> List[Lead]:
        """
        Get leads filtered by a specific quality issue type.
        """
        if issue == DataQualityIssue.MISSING_CONTACT:
            stmt = select(Lead).where(
                and_(
                    Lead.is_deleted == False,
                    or_(
                        Lead.phone.is_(None),
                        Lead.email.is_(None),
                        Lead.telegram_id.is_(None)
                    )
                )
            ).offset(offset).limit(limit)
        
        elif issue == DataQualityIssue.MISSING_NAME:
            stmt = select(Lead).where(
                and_(
                    Lead.is_deleted == False,
                    or_(Lead.full_name.is_(None), Lead.full_name == "")
                )
            ).offset(offset).limit(limit)
        
        elif issue == DataQualityIssue.STALE_LEAD:
            stale_cutoff = datetime.now(UTC) - timedelta(days=14)
            stmt = select(Lead).where(
                and_(
                    Lead.is_deleted == False,
                    Lead.updated_at < stale_cutoff,
                    Lead.stage.in_([ColdStage.NEW, ColdStage.CONTACTED])
                )
            ).offset(offset).limit(limit)
        
        elif issue == DataQualityIssue.NO_ACTIVITY:
            stmt = select(Lead).where(
                and_(
                    Lead.is_deleted == False,
                    Lead.message_count == 0
                )
            ).offset(offset).limit(limit)
        
        else:
            # Generic fallback
            stmt = select(Lead).where(Lead.is_deleted == False).offset(offset).limit(limit)
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
