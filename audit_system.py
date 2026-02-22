import asyncio
import sys
from datetime import datetime, UTC
from httpx import ASGITransport, AsyncClient

from app.core.database import AsyncSessionLocal
from app.models.lead import ColdStage, LeadSource, BusinessDomain
from app.models.user import User, UserRole
from app.repositories.user_repo import UserRepository
from app.core.config import settings

async def audit_system():
    print("🚀 Starting Complete System Audit...\n")
    
    # 1. Setup Admin User via directly to DB (so auth works)
    print("--- 1. Checking Database Connection & User Setup ---")
    async with AsyncSessionLocal() as session:
        user_repo = UserRepository(session)
        admin = await user_repo.get_by_telegram_id("test_admin_audit")
        if not admin:
            admin = User(
                telegram_id="test_admin_audit",
                username="AuditAdmin",
                full_name="Audit Admin",
                role=UserRole.ADMIN,
                is_active=True
            )
            session.add(admin)
            await session.commit()
            await session.refresh(admin)
        print("✅ Database connected. Admin user ready.")
        admin_id = admin.id

    from main import app
    headers = {"Authorization": f"Bearer {settings.API_SECRET_TOKEN}"}
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        
        # 2. Testing App Flow (Create -> Update -> Note -> Assign)
        print("\n--- 2. Testing Core Lead Flow (API) ---")
        
        # Create Lead
        resp = await client.post("/api/v1/leads", json={
            "source": LeadSource.MANUAL.value,
            "business_domain": BusinessDomain.FIRST.value,
            "telegram_id": "audit_test_123"
        }, headers=headers)
        
        assert resp.status_code == 201, f"Failed to create lead: {resp.text}"
        lead = resp.json()
        lead_id = lead["id"]
        print(f"✅ Lead created successfully (ID: {lead_id})")
        assert lead["stage"] == ColdStage.NEW.value, "Lead didn't start in NEW stage"
        
        # Test Stage Progression
        resp = await client.patch(f"/api/v1/leads/{lead_id}/stage", json={
            "stage": ColdStage.CONTACTED.value
        }, headers=headers)
        assert resp.status_code == 200, f"Stage transition failed: {resp.text}"
        print(f"✅ Lead transitioned sequentially to CONTACTED")
        
        # Test Invalid Progression Skip (Target -> Transferred from Contacted)
        resp = await client.patch(f"/api/v1/leads/{lead_id}/stage", json={
            "stage": ColdStage.TRANSFERRED.value
        }, headers=headers)
        assert resp.status_code == 400, "API allowed invalid stage skipping!"
        print(f"✅ API correctly blocked invalid stage skipping")
        
        # Test Adding Notes
        resp = await client.post(f"/api/v1/leads/{lead_id}/notes", json={
            "content": "This is an audit test note.",
            "note_type": "comment",
            "author_id": str(admin_id),
            "author_name": "AuditAdmin"
        }, headers=headers)
        assert resp.status_code == 201, f"Failed to add note: {resp.text}"
        note_id = resp.json()["id"]
        print(f"✅ Note added successfully")
        
        # Test Audit Logs 
        resp = await client.get(f"/api/v1/leads/{lead_id}/history", headers=headers)
        assert resp.status_code == 200
        history = resp.json()
        assert len(history) > 0, "No audit logs found!"
        print(f"✅ Audit logs captured transitions ({len(history)} records found)")
        
        # 3. Testing Automation & Transfers
        print("\n--- 3. Testing Automation & Sale Transfers ---")
        
        # Mock AI Analysis (requires AI service, bypassing via Manual DB injection for scale test)
        async with AsyncSessionLocal() as session:
            from app.models.lead import Lead
            from sqlalchemy import select
            db_lead = await session.execute(select(Lead).where(Lead.id == lead_id))
            db_lead = db_lead.scalar_one()
            
            db_lead.ai_score = 0.8
            db_lead.ai_recommendation = "High potential"
            db_lead.stage = ColdStage.QUALIFIED
            await session.commit()
            
        print("✅ Mocked Lead to QUALIFIED with AI Score 0.8")
        
        # Transfer
        resp = await client.post(f"/api/v1/leads/{lead_id}/transfer", params={
            "amount": 5000
        }, headers=headers)
        assert resp.status_code == 201, f"Failed to transfer to sales: {resp.text}"
        sale = resp.json()
        print(f"✅ Lead successfully transferred to Sales. Sale ID: {sale['id']}")
        
        # Verify Lead State is TRANSFERRED
        resp = await client.get(f"/api/v1/leads/{lead_id}", headers=headers)
        assert resp.json()["stage"] == ColdStage.TRANSFERRED.value, "Lead not marked as TRANSFERRED after sale created"
        print("✅ Lead marked as terminal TRANSFERRED")
        
        # 4. Cleanup
        print("\n--- 4. Data Cleanup Validation ---")
        resp = await client.delete(f"/api/v1/leads/{lead_id}", headers=headers)
        assert resp.status_code == 204, f"Failed to delete lead: {resp.text}"
        
        # Ensure it's gone
        resp = await client.get(f"/api/v1/leads/{lead_id}", headers=headers)
        assert resp.status_code == 404, "Lead was not actually deleted"
        print("✅ Delete endpoint successfully scrubbed lead & relationships")

    print("\n🎉 ALL CORE BUSINESS LOGIC CIRCUITS PASS! 🎉")


if __name__ == "__main__":
    try:
        asyncio.run(audit_system())
    except AssertionError as e:
        print(f"\n❌ AUDIT FAILED: {e}")
        sys.exit(1)
