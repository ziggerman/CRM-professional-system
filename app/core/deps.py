"""
FastAPI dependencies for dependency injection.
"""
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.repositories.lead_repo import LeadRepository
from app.repositories.sale_repo import SaleRepository
from app.repositories.user_repo import UserRepository
from app.services.lead_service import LeadService
from app.services.transfer_service import TransferService
from app.services.automation_service import AutomationService
from app.ai.ai_service import AIService


# Type aliases for cleaner dependency injection
DbSession = Annotated[AsyncSession, Depends(get_db)]


async def get_lead_repo(db: DbSession) -> LeadRepository:
    """Get LeadRepository instance."""
    return LeadRepository(db)


async def get_sale_repo(db: DbSession) -> SaleRepository:
    """Get SaleRepository instance."""
    return SaleRepository(db)


async def get_user_repo(db: DbSession) -> UserRepository:
    """Get UserRepository instance."""
    return UserRepository(db)


async def get_ai_service() -> AIService:
    """Get AIService instance."""
    return AIService()


async def get_lead_service(
    lead_repo: Annotated[LeadRepository, Depends(get_lead_repo)]
) -> LeadService:
    """Get LeadService instance."""
    return LeadService(lead_repo)


async def get_transfer_service(
    lead_repo: Annotated[LeadRepository, Depends(get_lead_repo)],
    sale_repo: Annotated[SaleRepository, Depends(get_sale_repo)],
    ai_service: Annotated[AIService, Depends(get_ai_service)],
) -> TransferService:
    """Get TransferService instance."""
    return TransferService(lead_repo, sale_repo, ai_service)


async def get_automation_service(
    lead_repo: Annotated[LeadRepository, Depends(get_lead_repo)],
    sale_repo: Annotated[SaleRepository, Depends(get_sale_repo)],
    user_repo: Annotated[UserRepository, Depends(get_user_repo)],
) -> AutomationService:
    """Get AutomationService instance."""
    return AutomationService(lead_repo, sale_repo, user_repo)
