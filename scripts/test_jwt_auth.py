import asyncio
import httpx
from app.core.security import get_password_hash
from app.core.database import AsyncSessionLocal
from app.models.user import User, UserRole
from sqlalchemy import select

async def setup_test_user():
    async with AsyncSessionLocal() as session:
        # Check if test user exists
        result = await session.execute(select(User).where(User.email == "test@ael.crm"))
        user = result.scalar_one_or_none()
        
        if not user:
            print("Creating test user...")
            user = User(
                full_name="Test Admin",
                email="test@ael.crm",
                hashed_password=get_password_hash("testpassword123"),
                role=UserRole.ADMIN,
                is_active=True
            )
            session.add(user)
            await session.commit()
            print("Test user created.")
        else:
            print("Test user already exists.")

async def test_login():
    url = "http://localhost:8000/api/v1/auth/login"
    login_data = {
        "email": "test@ael.crm",
        "password": "testpassword123"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # Login
            print(f"Attempting login to {url}...")
            response = await client.post(url, json=login_data)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                token_data = response.json()
                token = token_data["access_token"]
                print(f"Login successful. Token: {token[:20]}...")
                
                # Test protected endpoint
                protected_url = "http://localhost:8000/api/v1/health"
                print(f"Testing health check with token...")
                resp = await client.get(protected_url, headers={"Authorization": f"Bearer {token}"})
                print(f"Health Status: {resp.status_code}")
                print(resp.json())
            else:
                print(f"Login failed: {response.text}")
                
        except Exception as e:
            print(f"Error during test: {e}")

if __name__ == "__main__":
    # Note: main.py must be running for test_login to work
    asyncio.run(setup_test_user())
    print("\nTo test login, start the server and run this script against it.")
