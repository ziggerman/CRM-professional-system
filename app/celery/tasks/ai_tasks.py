"""
Celery tasks for AI operations.
"""
import logging
from celery import Task
from sqlalchemy.ext.asyncio import AsyncSession

from app.celery.config import celery_app
from app.core.database import AsyncSessionLocal
from app.models.lead import Lead
from app.repositories.lead_repo import LeadRepository
from app.services.transfer_service import TransferService
from app.ai.ai_service import AIService, AIServiceError
from app.repositories.sale_repo import SaleRepository

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task with database session management."""
    _db_session: AsyncSession | None = None
    
    @property
    def db_session(self) -> AsyncSession:
        if self._db_session is None:
            raise RuntimeError("Database session not initialized")
        return self._db_session
    
    def after_return(self, *args, **kwargs):
        """Close database session after task."""
        if self._db_session:
            import asyncio
            asyncio.run(self._db_session.close())


@celery_app.task(bind=True, base=DatabaseTask, max_retries=3, default_retry_delay=60)
def analyze_lead_task(self, lead_id: int) -> dict:
    """
    Async task to analyze a lead with AI.
    
    Args:
        lead_id: ID of the lead to analyze
        
    Returns:
        Dict with analysis result
    """
    import asyncio
    
    async def _analyze():
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            ai_service = AIService()
            
            # Get lead
            lead = await lead_repo.get_by_id(lead_id)
            if not lead:
                logger.error(f"Lead {lead_id} not found")
                return {"error": "Lead not found"}
            
            try:
                # Run AI analysis
                result = await ai_service.analyze_lead(lead)
                
                # Save result
                lead.ai_score = result.score
                lead.ai_recommendation = result.recommendation
                lead.ai_reason = result.reason
                await lead_repo.save(lead)
                await session.commit()
                
                logger.info(f"Lead {lead_id} analyzed successfully. Score: {result.score}")
                return {
                    "lead_id": lead_id,
                    "score": result.score,
                    "recommendation": result.recommendation,
                    "reason": result.reason,
                }
            except AIServiceError as e:
                logger.error(f"AI service error for lead {lead_id}: {e}")
                raise self.retry(exc=e)
    
    return asyncio.run(_analyze())


@celery_app.task(bind=True, base=DatabaseTask)
def batch_analyze_leads_task(self, lead_ids: list[int]) -> dict:
    """
    Batch analyze multiple leads.
    
    Args:
        lead_ids: List of lead IDs to analyze
        
    Returns:
        Dict with batch results
    """
    import asyncio
    
    async def _batch_analyze():
        results = []
        async with AsyncSessionLocal() as session:
            lead_repo = LeadRepository(session)
            ai_service = AIService()
            
            for lead_id in lead_ids:
                try:
                    lead = await lead_repo.get_by_id(lead_id)
                    if not lead:
                        results.append({"lead_id": lead_id, "error": "Not found"})
                        continue
                    
                    result = await ai_service.analyze_lead(lead)
                    
                    # Save result
                    lead.ai_score = result.score
                    lead.ai_recommendation = result.recommendation
                    lead.ai_reason = result.reason
                    await lead_repo.save(lead)
                    
                    results.append({
                        "lead_id": lead_id,
                        "score": result.score,
                        "recommendation": result.recommendation,
                    })
                except Exception as e:
                    logger.error(f"Error analyzing lead {lead_id}: {e}")
                    results.append({"lead_id": lead_id, "error": str(e)})
            
            await session.commit()
            return {"processed": len(results), "results": results}
    
    return asyncio.run(_batch_analyze())
