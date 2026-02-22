"""
Authentication API endpoints.
Step 3.1 — JWT Authentication
"""
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import verify_password, create_access_token, get_current_user
from app.repositories.user_repo import UserRepository
from app.models.user import User
from app.schemas.token import Token
from app.schemas.auth import LoginRequest, PasswordChangeRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and return JWT token.
    """
    user_repo = UserRepository(db)
    user = await user_repo.get_by_email(login_data.email)
    
    if not user or not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": str(user.id), "role": user.role.value if hasattr(user.role, 'value') else str(user.role)}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login/form", response_model=Token)
async def login_oauth2(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    OAuth2 compatible login (for Swagger UI 'Authorize' button).
    Expects username (email) and password in form-data.
    """
    return await login(LoginRequest(email=form_data.username, password=form_data.password), db)


@router.post("/setup-password")
async def setup_password(
    data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Set or change password for the current user.
    Enables login for users coming from Telegram.
    """
    from app.core.security import get_password_hash
    current_user.hashed_password = get_password_hash(data.password)
    await db.commit()
    return {"message": "Password updated successfully"}
