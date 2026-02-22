from sqlalchemy.ext.asyncio import AsyncSession
from app.models.ai_log import AIAnalysisLog

class AIRepo:
    """Repository for AI logging."""
    def __init__(self, db: AsyncSession):
        self.db = db

    async def log_analysis(self, log_entry: AIAnalysisLog) -> AIAnalysisLog:
        """Persist AI analysis log."""
        self.db.add(log_entry)
        await self.db.flush()
        return log_entry
