"""
Export API endpoints.
"""
from fastapi import APIRouter, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status
import csv
import io
from app.core.database import get_db
from app.models.lead import Lead
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.security import verify_api_token, require_role
from app.celery.tasks.export_tasks import export_leads_csv_task

router = APIRouter(dependencies=[Depends(verify_api_token)])


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def trigger_export(
    admin_id: int = Query(..., description="Telegram ID of the admin receiving the export"),
    role: str = Depends(require_role("admin"))
):
    """
    Trigger an asynchronous CSV export of CRM data.
    Only available to admins.
    """
    # Trigger Celery task
    task = export_leads_csv_task.delay(admin_id)
    return {"task_id": task.id, "status": "queued"}


@router.get("/download")
async def download_export(
    db: AsyncSession = Depends(get_db),
    role: str = Depends(require_role("admin"))
):
    """
    Direct streaming download of CRM data as CSV.
    Uses a generator to handle massive results without memory spikes.
    Step 6.1 — Business Logic / Scale
    """
    async def generate_csv():
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Lead ID", "Full Name", "Email", "Phone", "Source", "Stage", "Domain", 
            "Messages", "AI Score", "Created At"
        ])
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        # Stream leads from DB
        result = await db.stream(
            select(Lead).order_by(Lead.created_at.desc())
        )
        
        async for row in result:
            lead = row[0]
            writer.writerow([
                lead.id,
                lead.full_name or "N/A",
                lead.email or "N/A",
                lead.phone or "N/A",
                lead.source.value if hasattr(lead.source, 'value') else str(lead.source),
                lead.stage.value if hasattr(lead.stage, 'value') else str(lead.stage),
                lead.business_domain.value if lead.business_domain else "N/A",
                lead.message_count,
                lead.ai_score or 0.0,
                lead.created_at.strftime("%Y-%m-%d %H:%M:%S")
            ])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    return StreamingResponse(
        generate_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=crm_export.csv"}
    )
