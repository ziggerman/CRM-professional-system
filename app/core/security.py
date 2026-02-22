"""
API Security module.
Step 3.1 — JWT Authentication
"""
from datetime import datetime, UTC, timedelta
from typing import Optional, Annotated

from fastapi import Depends, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from app.core.config import settings
from app.core.database import get_db
from app.models.user import User, UserRole
from app.schemas.token import TokenData

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer authentication scheme
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed one."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a hash for a password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user from JWT.
    Step 3.1 implementation.
    """
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # Fallback to static token for admin access
    if settings.API_SECRET_TOKEN and token == settings.API_SECRET_TOKEN:
        return User(id=0, username="static_admin", role=UserRole.ADMIN, is_active=True)

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=int(user_id))
    except (JWTError, ValueError):
        raise credentials_exception
    
    from app.repositories.user_repo import UserRepository
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(token_data.user_id)
    
    if user is None or not user.is_active:
        raise credentials_exception
    return user

def verify_api_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Backwards compatibility for static API tokens (internal bot).
    Also allows valid JWTs if we want to support both in same endpoints.
    """
    if credentials.credentials == settings.API_SECRET_TOKEN:
        return credentials.credentials
        
    # Attempt to decode as JWT for mixed support
    try:
        jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return credentials.credentials
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Role Hierarchy Definition
_HIERARCHY_VALUES = {
    UserRole.AGENT: 1,
    UserRole.MANAGER: 2,
    UserRole.ADMIN: 3,
    UserRole.SUPER_ADMIN: 4
}
# Expand to include string keys for robustness
ROLE_HIERARCHY = {}
for role, level in _HIERARCHY_VALUES.items():
    ROLE_HIERARCHY[role] = level
    ROLE_HIERARCHY[role.value.upper()] = level

def require_role(min_role: UserRole):
    """
    Dependency factory to enforce minimum role requirements using JWT user.
    """
    async def role_checker(current_user: User = Depends(get_current_user)):
        user_role = current_user.role
        
        user_level = ROLE_HIERARCHY.get(user_role, 0)
        
        role_key = min_role.value if hasattr(min_role, "value") else str(min_role).upper()
        required_level = ROLE_HIERARCHY.get(role_key, 99)
        
        if user_level < required_level:
            role_val = min_role.value if hasattr(min_role, "value") else str(min_role)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation forbidden. Required role: {role_val}"
            )
        return current_user
        
    return role_checker
