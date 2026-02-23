"""
Authentication API endpoints.
Step 3.1 — JWT Authentication
Step 4 — Enhanced Security: Refresh Tokens, RBAC, Rate Limiting
"""
import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import verify_password, create_access_token, create_refresh_token, verify_refresh_token, get_current_user
from app.core.config import settings
from app.repositories.user_repo import UserRepository
from app.models.user import User
from app.schemas.token import Token
from app.schemas.auth import LoginRequest, PasswordChangeRequest, RefreshTokenRequest
from app.api.rate_limit import brute_force_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


async def _check_brute_force(request: Request, email: str) -> None:
    """
    Check for brute-force attacks on login endpoint.
    Raises HTTPException if too many failed attempts detected.
    """
    if not brute_force_store:
        return
    
    client_ip = request.client.host if request.client else "unknown"
    key = f"login_failed:{email}:{client_ip}"
    
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        
        attempts = await redis_client.get(key)
        if attempts and int(attempts) >= settings.BRUTE_FORCE_MAX_ATTEMPTS:
            # Check if still locked
            ttl = await redis_client.ttl(key)
            if ttl > 0:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Too many failed login attempts. Please try again in {ttl // 60 + 1} minutes.",
                    headers={"Retry-After": str(ttl)}
                )
    except Exception as e:
        logger.warning(f"Brute-force check failed: {e}")


async def _record_failed_attempt(request: Request, email: str) -> None:
    """Record failed login attempt for brute-force protection."""
    if not brute_force_store:
        return
    
    client_ip = request.client.host if request.client else "unknown"
    key = f"login_failed:{email}:{client_ip}"
    
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        
        pipe = redis_client.pipeline()
        pipe.incr(key)
        # Set expiry only on first attempt
        current = await redis_client.get(key)
        if current and int(current) == 1:
            await redis_client.expire(key, settings.BRUTE_FORCE_LOCKOUT_MINUTES * 60)
        await pipe.execute()
    except Exception as e:
        logger.warning(f"Failed to record login attempt: {e}")


async def _clear_failed_attempts(request: Request, email: str) -> None:
    """Clear failed login attempts after successful login."""
    if not brute_force_store:
        return
    
    client_ip = request.client.host if request.client else "unknown"
    key = f"login_failed:{email}:{client_ip}"
    
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        await redis_client.delete(key)
    except Exception as e:
        logger.warning(f"Failed to clear login attempts: {e}")


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and return JWT token.
    Step 4 — Includes brute-force protection.
    """
    # Check brute-force protection
    await _check_brute_force(request, login_data.email)
    
    user_repo = UserRepository(db)
    user = await user_repo.get_by_email(login_data.email)
    
    if not user or not user.hashed_password:
        await _record_failed_attempt(request, login_data.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not verify_password(login_data.password, user.hashed_password):
        await _record_failed_attempt(request, login_data.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.is_active:
        await _record_failed_attempt(request, login_data.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Clear failed attempts on successful login
    await _clear_failed_attempts(request, login_data.email)

    # Create access and refresh tokens
    access_token = create_access_token(
        data={"sub": str(user.id), "role": user.role.value if hasattr(user.role, 'value') else str(user.role)}
    )
    refresh_token = create_refresh_token(
        data={"sub": str(user.id), "role": user.role.value if hasattr(user.role, 'value') else str(user.role)}
    )
    
    logger.info(f"User {user.id} logged in successfully")
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    Step 4 — New endpoint for token refresh.
    """
    # Verify refresh token
    payload = verify_refresh_token(request.refresh_token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(int(user_id))
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create new access and refresh tokens
    access_token = create_access_token(
        data={"sub": str(user.id), "role": user.role.value if hasattr(user.role, 'value') else str(user.role)}
    )
    new_refresh_token = create_refresh_token(
        data={"sub": str(user.id), "role": user.role.value if hasattr(user.role, 'value') else str(user.role)}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@router.post("/login/form", response_model=Token)
async def login_oauth2(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    OAuth2 compatible login (for Swagger UI 'Authorize' button).
    Expects username (email) and password in form-data.
    """
    return await login(request, LoginRequest(email=form_data.username, password=form_data.password), db)


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
