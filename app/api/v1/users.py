from fastapi import APIRouter, Depends, HTTPException, Query
from starlette import status

from app.core.database import get_db
from app.repositories.user_repo import UserRepository
from app.core.security import get_current_user, require_role, get_password_hash
from app.models.user import User, UserRole
from app.schemas.user import UserResponse, UserCreate
from sqlalchemy.ext.asyncio import AsyncSession

# Standard naming used by leads.py
router = APIRouter()

@router.get("/me", response_model=UserResponse)
async def get_my_profile(
    current_user: User = Depends(get_current_user)
):
    """Fetch the currently authenticated user's profile."""
    return current_user

@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    data: UserCreate,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: AsyncSession = Depends(get_db)
):
    """Create a new user/manager. Admin only."""
    repo = UserRepository(db)
    existing = await repo.get_by_email(data.email)
    if existing:
        raise HTTPException(status_code=400, detail="User with this email already exists")
    
    new_user = User(
        email=data.email,
        hashed_password=get_password_hash(data.password),
        full_name=data.full_name,
        username=data.username,
        role=data.role,
        is_active=data.is_active
    )
    user = await repo.create(new_user)
    await db.commit()
    return user

@router.get("/legacy/by-telegram", response_model=UserResponse)
async def get_user_by_telegram_id(
    telegram_id: str = Query(..., description="The Telegram ID of the user"),
    db: AsyncSession = Depends(get_db)
):
    """Legacy fetch user logic based on telegram_id for Bot."""
    repo = UserRepository(db)
    user = await repo.get_by_telegram_id(telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
