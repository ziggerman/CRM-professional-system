import asyncio
import sys
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal
from app.models.lead import ColdStage, LeadSource
from app.services.lead_service import LeadService
from app.repositories.lead_repo import LeadRepository
from app.repositories.history_repo import HistoryRepository
from app.schemas.lead import LeadCreate
from app.celery.tasks.lead_tasks import process_stale_leads_task
from app.core.config import settings

async def main():
    print("=== 1. Setting up Database Data ===")
    lead_id = None
    async with AsyncSessionLocal() as session:
        lead_repo = LeadRepository(session)
        history_repo = HistoryRepository(session)
        svc = LeadService(lead_repo, history_repo)
        
        # Create
        data = LeadCreate(source=LeadSource.MANUAL, telegram_id="test_user_123")
        lead = await svc.create_lead(data)
        lead_id = lead.id
        print(f"✅ Created Lead ID: {lead.id}")
        
        # Transition
        lead = await svc.transition_stage(lead, ColdStage.CONTACTED)
        print(f"✅ Transitioned Lead {lead.id} to {ColdStage.CONTACTED.value}")
        
        # Transition again
        lead = await svc.transition_stage(lead, ColdStage.QUALIFIED)
        print(f"✅ Transitioned Lead {lead.id} to {ColdStage.QUALIFIED.value}")
        
        await session.commit()

    from main import app
    from httpx import ASGITransport
    print("\n=== 2. Testing API Security & Audit Logs ===")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Test Unauthenticated
        resp = await client.get(f"/api/v1/leads/{lead_id}/history")
        if resp.status_code == 401:
            print("✅ Unauthenticated API access blocked (401)")
        else:
            print(f"❌ Failed Unauthenticated Test, expected 401 got {resp.status_code}")
            
        # Test Authenticated
        headers = {"Authorization": f"Bearer {settings.API_SECRET_TOKEN}"}
        resp = await client.get(f"/api/v1/leads/{lead_id}/history", headers=headers)
        if resp.status_code == 200:
            history = resp.json()
            print(f"✅ Authenticated API access successful (200). Found {len(history)} history records.")
            for h in history:
                print(f"  - [{h['created_at']}] {h['old_stage']} -> {h['new_stage']} by {h['changed_by']}")
        else:
            print(f"❌ Failed Authenticated Test, expected 200 got {resp.status_code}")
            print(resp.text)

    print("\n=== 3. Testing Celery Tasks ===")
    print("Triggering process_stale_leads_task...")
    try:
        # Just run the inner coroutine directly or the sync wrapper
        result = process_stale_leads_task()
        print(f"✅ Celery Task Executed. Result: {result}")
    except Exception as e:
        print(f"❌ Celery Task Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
