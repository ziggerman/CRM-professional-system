"""
KPI Service - Analytics and metrics calculation for the dashboard.
Implements Step 7: KPI Dashboard with conversion per stage, median response time,
win rate by source/domain/agent, lead aging & overdue, and historical trends.
"""
import logging
from datetime import datetime, timedelta, UTC
from typing import Optional
from collections import defaultdict

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.lead import Lead, ColdStage, LeadSource, BusinessDomain
from app.models.sale import Sale, SaleStage
from app.models.user import User
from app.models.activity import LeadActivity

logger = logging.getLogger(__name__)


class KPIService:
    """Service for calculating KPIs and analytics."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # ──────────────────────────────────────────────
    # Conversion per Stage
    # ──────────────────────────────────────────────
    
    async def get_conversion_per_stage(self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> dict:
        """
        Calculate conversion rates at each stage of the pipeline.
        
        Returns:
            Dict with leads at each stage and conversion rates
        """
        conditions = [Lead.is_deleted == False]
        if start_date:
            conditions.append(Lead.created_at >= start_date)
        if end_date:
            conditions.append(Lead.created_at <= end_date)
        
        stmt = select(Lead).where(and_(*conditions))
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())
        
        total = len(leads)
        if total == 0:
            return {"total": 0, "stages": {}, "conversion_rates": {}}
        
        # Count by stage
        stage_counts = defaultdict(int)
        for lead in leads:
            stage_counts[lead.stage.value] += 1
        
        # Calculate cumulative conversion (leads that passed through each stage)
        # NEW -> CONTACTED -> QUALIFIED -> TRANSFERRED
        stages_order = ["new", "contacted", "qualified", "transferred"]
        cumulative = 0
        conversion_rates = {}
        
        for stage in stages_order:
            count = stage_counts.get(stage, 0)
            cumulative += count
            if total > 0:
                conversion_rates[stage] = round(cumulative / total * 100, 2)
            else:
                conversion_rates[stage] = 0
        
        return {
            "total": total,
            "stages": dict(stage_counts),
            "conversion_rates": conversion_rates,
        }
    
    # ──────────────────────────────────────────────
    # Median Response Time
    # ──────────────────────────────────────────────
    
    async def get_median_response_time(self) -> dict:
        """
        Calculate median time from lead creation to first response.
        
        Returns:
            Dict with median response time in hours and breakdown by stage
        """
        # Get leads that have been contacted (first_response_at is set)
        stmt = select(Lead).where(
            and_(
                Lead.is_deleted == False,
                Lead.first_response_at.isnot(None),
            )
        )
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())
        
        if not leads:
            return {
                "median_hours": None,
                "total_responded": 0,
                "by_stage": {}
            }
        
        # Calculate response times
        response_times = []
        by_stage = defaultdict(list)
        
        for lead in leads:
            created = lead.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            
            responded = lead.first_response_at
            if responded and created:
                hours = (responded - created).total_seconds() / 3600
                response_times.append(hours)
                by_stage[lead.stage.value].append(hours)
        
        if not response_times:
            return {
                "median_hours": None,
                "total_responded": 0,
                "by_stage": {}
            }
        
        # Calculate median
        response_times.sort()
        n = len(response_times)
        if n % 2 == 0:
            median = (response_times[n//2 - 1] + response_times[n//2]) / 2
        else:
            median = response_times[n//2]
        
        # Calculate median by stage
        stage_medians = {}
        for stage, times in by_stage.items():
            if times:
                times.sort()
                n_stage = len(times)
                if n_stage % 2 == 0:
                    stage_medians[stage] = round((times[n_stage//2 - 1] + times[n_stage//2]) / 2, 2)
                else:
                    stage_medians[stage] = round(times[n_stage//2], 2)
        
        return {
            "median_hours": round(median, 2),
            "total_responded": len(response_times),
            "avg_hours": round(sum(response_times) / len(response_times), 2),
            "min_hours": round(min(response_times), 2),
            "max_hours": round(max(response_times), 2),
            "by_stage": stage_medians,
        }
    
    # ──────────────────────────────────────────────
    # Win Rate by Source/Domain/Agent
    # ──────────────────────────────────────────────
    
    async def get_win_rate_by_source(self) -> dict:
        """
        Calculate win rate (transferred / total) grouped by lead source.
        """
        stmt = select(Lead).where(Lead.is_deleted == False)
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())
        
        if not leads:
            return {}
        
        by_source = defaultdict(lambda: {"total": 0, "won": 0})
        
        for lead in leads:
            source = lead.source.value
            by_source[source]["total"] += 1
            if lead.stage == ColdStage.TRANSFERRED:
                by_source[source]["won"] += 1
        
        win_rates = {}
        for source, counts in by_source.items():
            total = counts["total"]
            won = counts["won"]
            win_rates[source] = {
                "total": total,
                "won": won,
                "win_rate": round(won / total * 100, 2) if total > 0 else 0
            }
        
        return win_rates
    
    async def get_win_rate_by_domain(self) -> dict:
        """
        Calculate win rate grouped by business domain.
        """
        stmt = select(Lead).where(
            and_(
                Lead.is_deleted == False,
                Lead.business_domain.isnot(None)
            )
        )
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())
        
        if not leads:
            return {}
        
        by_domain = defaultdict(lambda: {"total": 0, "won": 0})
        
        for lead in leads:
            domain = lead.business_domain.value if lead.business_domain else "UNKNOWN"
            by_domain[domain]["total"] += 1
            if lead.stage == ColdStage.TRANSFERRED:
                by_domain[domain]["won"] += 1
        
        win_rates = {}
        for domain, counts in by_domain.items():
            total = counts["total"]
            won = counts["won"]
            win_rates[domain] = {
                "total": total,
                "won": won,
                "win_rate": round(won / total * 100, 2) if total > 0 else 0
            }
        
        return win_rates
    
    async def get_win_rate_by_agent(self) -> dict:
        """
        Calculate win rate grouped by assigned agent.
        """
        stmt = select(Lead).where(
            and_(
                Lead.is_deleted == False,
                Lead.assigned_to_id.isnot(None)
            )
        )
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())
        
        if not leads:
            return {}
        
        # Get user names
        user_ids = set(l.assigned_to_id for l in leads if l.assigned_to_id)
        users_stmt = select(User).where(User.id.in_(user_ids))
        users_result = await self.db.execute(users_stmt)
        users = users_result.scalars().all()
        user_names = {u.id: u.full_name for u in users}
        
        by_agent = defaultdict(lambda: {"total": 0, "won": 0})
        
        for lead in leads:
            agent_id = lead.assigned_to_id
            if agent_id:
                by_agent[agent_id]["total"] += 1
                if lead.stage == ColdStage.TRANSFERRED:
                    by_agent[agent_id]["won"] += 1
        
        win_rates = {}
        for agent_id, counts in by_agent.items():
            total = counts["total"]
            won = counts["won"]
            win_rates[user_names.get(agent_id, f"Agent {agent_id}")] = {
                "user_id": agent_id,
                "total": total,
                "won": won,
                "win_rate": round(won / total * 100, 2) if total > 0 else 0
            }
        
        return win_rates
    
    # ──────────────────────────────────────────────
    # Lead Aging & Overdue
    # ──────────────────────────────────────────────
    
    async def get_lead_aging(self) -> dict:
        """
        Calculate lead aging statistics - how long leads stay in each stage.
        
        Returns:
            Dict with average days in stage and overdue lead counts
        """
        stmt = select(Lead).where(Lead.is_deleted == False)
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())
        
        if not leads:
            return {"average_days": {}, "overdue_leads": 0}
        
        now = datetime.now(UTC)
        
        # Group by stage and calculate days
        stage_days = defaultdict(list)
        overdue_count = 0
        
        for lead in leads:
            # Calculate days in current stage
            updated = lead.updated_at
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            
            days = (now - updated).days
            stage_days[lead.stage.value].append(days)
            
            # Check if overdue (SLA breached)
            if lead.is_overdue:
                overdue_count += 1
        
        # Calculate averages
        average_days = {}
        for stage, days_list in stage_days.items():
            if days_list:
                average_days[stage] = round(sum(days_list) / len(days_list), 1)
        
        # Count leads by age buckets
        age_buckets = {
            "0-7_days": 0,
            "8-14_days": 0,
            "15-30_days": 0,
            "31-60_days": 0,
            "60+_days": 0,
        }
        
        for lead in leads:
            if lead.stage in (ColdStage.TRANSFERRED, ColdStage.LOST):
                continue
            
            updated = lead.updated_at
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            
            days = (now - updated).days
            
            if days <= 7:
                age_buckets["0-7_days"] += 1
            elif days <= 14:
                age_buckets["8-14_days"] += 1
            elif days <= 30:
                age_buckets["15-30_days"] += 1
            elif days <= 60:
                age_buckets["31-60_days"] += 1
            else:
                age_buckets["60+_days"] += 1
        
        return {
            "average_days_by_stage": average_days,
            "overdue_leads": overdue_count,
            "age_distribution": age_buckets,
            "total_active": sum(1 for l in leads if l.stage not in (ColdStage.TRANSFERRED, ColdStage.LOST)),
        }
    
    # ──────────────────────────────────────────────
    # Historical Trends (Week/Month)
    # ──────────────────────────────────────────────
    
    async def get_weekly_trends(self, weeks: int = 12) -> dict:
        """
        Get weekly trend data for the last N weeks.
        
        Returns:
            Dict with weekly metrics (new leads, converted, lost, response times)
        """
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(weeks=weeks)
        
        stmt = select(Lead).where(
            and_(
                Lead.is_deleted == False,
                Lead.created_at >= start_date,
                Lead.created_at <= end_date
            )
        )
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())
        
        # Group by week
        weekly_data = defaultdict(lambda: {
            "new_leads": 0,
            "converted": 0,
            "lost": 0,
            "responded": 0,
            "response_times": []
        })
        
        for lead in leads:
            # Get week number (ISO week)
            week_key = lead.created_at.strftime("%Y-W%W")
            
            weekly_data[week_key]["new_leads"] += 1
            
            if lead.stage == ColdStage.TRANSFERRED:
                weekly_data[week_key]["converted"] += 1
            
            if lead.stage == ColdStage.LOST:
                weekly_data[week_key]["lost"] += 1
            
            if lead.first_response_at:
                weekly_data[week_key]["responded"] += 1
                created = lead.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                hours = (lead.first_response_at - created).total_seconds() / 3600
                weekly_data[week_key]["response_times"].append(hours)
        
        # Calculate rates and averages
        trends = []
        for week_key in sorted(weekly_data.keys()):
            data = weekly_data[week_key]
            response_times = data["response_times"]
            
            trends.append({
                "week": week_key,
                "new_leads": data["new_leads"],
                "converted": data["converted"],
                "lost": data["lost"],
                "responded": data["responded"],
                "conversion_rate": round(data["converted"] / data["new_leads"] * 100, 2) if data["new_leads"] > 0 else 0,
                "avg_response_hours": round(sum(response_times) / len(response_times), 2) if response_times else None,
            })
        
        return {"weeks": trends}
    
    async def get_monthly_trends(self, months: int = 12) -> dict:
        """
        Get monthly trend data for the last N months.
        """
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=months * 30)
        
        stmt = select(Lead).where(
            and_(
                Lead.is_deleted == False,
                Lead.created_at >= start_date,
                Lead.created_at <= end_date
            )
        )
        result = await self.db.execute(stmt)
        leads = list(result.scalars().all())
        
        # Group by month
        monthly_data = defaultdict(lambda: {
            "new_leads": 0,
            "converted": 0,
            "lost": 0,
            "responded": 0,
            "response_times": [],
            "revenue": 0
        })
        
        # Also get sales data for revenue
        sales_stmt = select(Sale).where(
            and_(
                Sale.created_at >= start_date,
                Sale.created_at <= end_date,
                Sale.stage == SaleStage.PAID
            )
        )
        sales_result = await self.db.execute(sales_stmt)
        sales = list(sales_result.scalars().all())
        
        for sale in sales:
            month_key = sale.created_at.strftime("%Y-%m")
            monthly_data[month_key]["revenue"] += sale.amount or 0
        
        for lead in leads:
            month_key = lead.created_at.strftime("%Y-%m")
            
            monthly_data[month_key]["new_leads"] += 1
            
            if lead.stage == ColdStage.TRANSFERRED:
                monthly_data[month_key]["converted"] += 1
            
            if lead.stage == ColdStage.LOST:
                monthly_data[month_key]["lost"] += 1
            
            if lead.first_response_at:
                monthly_data[month_key]["responded"] += 1
                created = lead.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                hours = (lead.first_response_at - created).total_seconds() / 3600
                monthly_data[month_key]["response_times"].append(hours)
        
        # Build trends
        trends = []
        for month_key in sorted(monthly_data.keys()):
            data = monthly_data[month_key]
            response_times = data["response_times"]
            
            trends.append({
                "month": month_key,
                "new_leads": data["new_leads"],
                "converted": data["converted"],
                "lost": data["lost"],
                "responded": data["responded"],
                "conversion_rate": round(data["converted"] / data["new_leads"] * 100, 2) if data["new_leads"] > 0 else 0,
                "avg_response_hours": round(sum(response_times) / len(response_times), 2) if response_times else None,
                "revenue": data["revenue"],
            })
        
        return {"months": trends}
    
    # ──────────────────────────────────────────────
    # Complete KPI Dashboard Data
    # ──────────────────────────────────────────────
    
    async def get_complete_kpi_dashboard(self) -> dict:
        """
        Get all KPI data in one call for the dashboard.
        """
        conversion = await self.get_conversion_per_stage()
        response_time = await self.get_median_response_time()
        win_source = await self.get_win_rate_by_source()
        win_domain = await self.get_win_rate_by_domain()
        win_agent = await self.get_win_rate_by_agent()
        aging = await self.get_lead_aging()
        weekly = await self.get_weekly_trends()
        monthly = await self.get_monthly_trends()
        
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "conversion_per_stage": conversion,
            "median_response_time": response_time,
            "win_rate_by_source": win_source,
            "win_rate_by_domain": win_domain,
            "win_rate_by_agent": win_agent,
            "lead_aging": aging,
            "weekly_trends": weekly,
            "monthly_trends": monthly,
        }
