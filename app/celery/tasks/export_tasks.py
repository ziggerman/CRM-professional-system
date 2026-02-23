"""
Celery tasks for data export.
"""
import logging
import csv
import io
import asyncio
from datetime import datetime
from aiogram import Bot
from aiogram.types import BufferedInputFile

from app.celery.config import celery_app
from app.core.database import AsyncSessionLocal
from app.core.config import settings
from app.models.lead import Lead
from app.models.sale import Sale
from sqlalchemy import select
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)


@celery_app.task
def export_leads_csv_task(admin_id: int) -> dict:
    """
    Generate a CSV export of all leads and sales, and send it to the admin via Telegram.
    """
    async def _export():
        async with AsyncSessionLocal() as session:
            # Fetch all leads with their associated sales
            result = await session.execute(
                select(Lead).options(selectinload(Lead.sale)).order_by(Lead.created_at.desc())
            )
            leads = result.scalars().all()
            
            if not leads:
                return {"status": "empty", "message": "No leads to export"}

            # Create CSV in memory
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow([
                "Lead ID", "Telegram ID", "Source", "Stage", "Domain", 
                "Messages", "AI Score", "Created At", 
                "Has Sale", "Sale Stage", "Sale Amount"
            ])
            
            # Data rows
            for lead in leads:
                has_sale = lead.sale is not None
                writer.writerow([
                    lead.id,
                    lead.telegram_id or "N/A",
                    lead.source.value if hasattr(lead.source, 'value') else str(lead.source),
                    lead.stage.value if hasattr(lead.stage, 'value') else str(lead.stage),
                    lead.business_domain.value if lead.business_domain else "N/A",
                    lead.message_count,
                    lead.ai_score or 0.0,
                    lead.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "Yes" if has_sale else "No",
                    lead.sale.stage.value if has_sale else "N/A",
                    lead.sale.amount / 100 if has_sale and lead.sale.amount else 0.0
                ])
            
            csv_data = output.getvalue().encode('utf-8')
            output.close()
            
            # Send via Telegram
            if settings.TELEGRAM_BOT_TOKEN:
                bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
                try:
                    input_file = BufferedInputFile(
                        csv_data, 
                        filename=f"crm_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
                    await bot.send_document(
                        admin_id, 
                        input_file, 
                        caption=f"📊 <b>CRM Data Export</b>\n\nTotal leads: {len(leads)}\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        parse_mode="HTML"
                    )
                except Exception as e:
                    logger.error(f"Failed to send export to admin {admin_id}: {e}")
                    return {"status": "error", "message": str(e)}
                finally:
                    await bot.session.close()
            
            return {"status": "success", "leads_exported": len(leads)}

    return asyncio.run(_export())
