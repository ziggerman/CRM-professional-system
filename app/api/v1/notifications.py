"""
Notifications and Bot interactions API.
Step 8.3 & 8.4
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.notification_service import NotificationService
from app.core.security import require_role
from app.models.user import UserRole

router = APIRouter(prefix="/notifications", tags=["notifications"])


class BroadcastRequest(BaseModel):
    """Request for admin broadcast."""
    text: str


@router.post("/broadcast")
async def admin_broadcast(
    request: BroadcastRequest,
    db: AsyncSession = Depends(get_db),
    admin = Depends(require_role(UserRole.ADMIN))
):
    """
    Broadcast a message to all active managers/admins (Step 8.3).
    """
    notif_svc = NotificationService()
    count = await notif_svc.notify_all_managers(request.text, db)
    await notif_svc.close()
    
    return {"message": f"Broadcast sent to {count} users", "count": count}


@router.get("/lead/{lead_id}/actions")
async def get_lead_quick_actions(
    lead_id: int,
    admin = Depends(require_role(UserRole.AGENT))
):
    """
    Get quick actions for a lead (for Bot interactive buttons) (Step 8.4).
    """
    # In a real implementation, this could return structured data for aiogram Keyboards
    return {
        "lead_id": lead_id,
        "actions": [
            {"label": "📞 Call", "callback": f"lead_call:{lead_id}"},
            {"label": "📝 Add Note", "callback": f"lead_note:{lead_id}"},
            {"label": "🚀 Transfer", "callback": f"lead_transfer:{lead_id}"},
            {"label": "❌ Lost", "callback": f"lead_lost:{lead_id}"}
        ]
    }
