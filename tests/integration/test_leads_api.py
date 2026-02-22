import pytest
from httpx import AsyncClient
from app.models.lead import ColdStage, LeadSource

@pytest.mark.asyncio
async def test_create_lead(client: AsyncClient, auth_token: str):
    """Test creating a lead via API."""
    data = {
        "telegram_id": "999888777",
        "full_name": "Integration Test Lead",
        "source": LeadSource.MESSAGE.value,
        "phone": "+1234567890",
        "email": "integration@test.com"
    }
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    response = await client.post("/api/v1/leads", json=data, headers=headers)
    assert response.status_code == 201
    res_data = response.json()
    assert res_data["full_name"] == data["full_name"]
    assert res_data["stage"] == ColdStage.NEW.value

@pytest.mark.asyncio
async def test_list_leads_with_pagination(client: AsyncClient, auth_token: str):
    """Test listing leads with pagination."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = await client.get("/api/v1/leads?page=1&page_size=10", headers=headers)
    assert response.status_code == 200
    res_data = response.json()
    assert "items" in res_data
    assert "total" in res_data

@pytest.mark.asyncio
async def test_bulk_update_stage(client: AsyncClient, auth_token: str):
    """Test bulk update operation."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    # 1. Create a lead
    create_resp = await client.post("/api/v1/leads", json={
        "telegram_id": "111", "full_name": "L1", "source": "MESSAGE"
    }, headers=headers)
    lead_id = create_resp.json()["id"]
    
    # 2. Bulk update
    bulk_data = {
        "lead_ids": [lead_id],
        "action": "update_stage",
        "stage": ColdStage.CONTACTED.value
    }
    response = await client.post("/api/v1/leads/bulk", json=bulk_data, headers=headers)
    assert response.status_code == 200
    
    # 3. Verify the update
    get_resp = await client.get(f"/api/v1/leads/{lead_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["stage"] == ColdStage.CONTACTED.value
