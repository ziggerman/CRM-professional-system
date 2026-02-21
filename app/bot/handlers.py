"""
Telegram Bot handlers - With My Leads feature.
"""
import logging
from typing import Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery

from app.bot.config import bot_settings
from app.bot.keyboards import (
    get_source_keyboard,
    get_domain_keyboard,
    get_lead_detail_keyboard,
    get_main_menu_keyboard,
    get_start_keyboard,
    get_menu_keyboard,
    get_settings_keyboard,
    get_quick_actions_keyboard,
    get_leads_category_keyboard,
    get_stage_subcategories_keyboard,
    get_source_subcategories_keyboard,
    get_domain_subcategories_keyboard,
    get_edit_stage_keyboard,
    get_edit_source_keyboard,
    get_edit_domain_keyboard,
    get_confirm_delete_keyboard,
)
from app.bot.states import LeadCreationState

logger = logging.getLogger(__name__)

router = Router()
bot: Optional[Bot] = None


def get_bot() -> Bot:
    global bot
    if bot is None:
        bot = Bot(token=bot_settings.TELEGRAM_BOT_TOKEN)
    return bot


# API Functions
async def create_lead_via_api(source: str, domain: str = None, telegram_id: int = None) -> dict:
    import httpx
    
    url = "http://localhost:8000/api/v1/leads"
    data = {"source": source, "telegram_id": str(telegram_id) if telegram_id else None}
    if domain:
        data["business_domain"] = domain
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=data, timeout=10.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"API error: {e}")
    return None


async def get_leads_via_api(stage: str = None, source: str = None, domain: str = None, assigned_to_id: int = None) -> list:
    import httpx
    
    params = {}
    if stage:
        params["stage"] = stage
    if source:
        params["source"] = source
    if domain:
        params["business_domain"] = domain
    if assigned_to_id:
        params["assigned_to_id"] = assigned_to_id
    
    url = "http://localhost:8000/api/v1/leads"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("items", [])
        except Exception as e:
            logger.error(f"API error: {e}")
    return []


async def get_lead_by_id_via_api(lead_id: int) -> dict:
    import httpx
    
    url = f"http://localhost:8000/api/v1/leads/{lead_id}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"API error: {e}")
    return None


async def update_lead_via_api(lead_id: int, data: dict) -> dict:
    import httpx
    
    url = f"http://localhost:8000/api/v1/leads/{lead_id}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.patch(url, json=data, timeout=10.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"API error: {e}")
    return None


async def update_lead_stage_via_api(lead_id: int, stage: str) -> dict:
    import httpx
    
    url = f"http://localhost:8000/api/v1/leads/{lead_id}/stage"
    data = {"stage": stage}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.patch(url, json=data, timeout=10.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"API error: {e}")
    return None


async def delete_lead_via_api(lead_id: int) -> bool:
    import httpx
    
    url = f"http://localhost:8000/api/v1/leads/{lead_id}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(url, timeout=10.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API error: {e}")
    return False


# Command Handlers

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    user_id = message.from_user.id
    is_admin = user_id in bot_settings.TELEGRAM_ADMIN_IDS
    
    if is_admin:
        await message.answer("=== CRM BOT ===\n\nWelcome! Choose action:", reply_markup=get_start_keyboard())
    else:
        await message.answer("CRM Bot\nContact admin for access.")


@router.message(Command("menu"))
@router.message(F.text == "Menu")
async def cmd_menu(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("=== MAIN MENU ===\n\nChoose action:", reply_markup=get_menu_keyboard())


@router.message(Command("settings"))
@router.message(F.text == "Settings")
async def cmd_settings(message: Message, state: FSMContext):
    await message.answer("=== SETTINGS ===\n\nAI: gpt-4o-mini\nMin score: 0.6", reply_markup=get_settings_keyboard())


@router.message(F.text == "Leads")
async def cmd_leads(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("=== LEADS ===\n\nSelect section:", reply_markup=get_leads_category_keyboard())


@router.message(F.text == "Stats")
async def cmd_statistics(message: Message, state: FSMContext):
    leads = await get_leads_via_api()
    total = len(leads)
    new_count = sum(1 for l in leads if l.get("stage") == "new")
    contacted = sum(1 for l in leads if l.get("stage") == "contacted")
    qualified = sum(1 for l in leads if l.get("stage") == "qualified")
    transferred = sum(1 for l in leads if l.get("stage") == "transferred")
    lost = sum(1 for l in leads if l.get("stage") == "lost")
    
    await message.answer(
        f"=== STATS ===\n\nTotal: {total}\nNew: {new_count}\nContacted: {contacted}\nQualified: {qualified}\nTransferred: {transferred}\nLost: {lost}\n\nConversion: {round(transferred/total*100, 1) if total > 0 else 0}%"
    )


@router.message(F.text == "NewLead")
async def cmd_new_lead(message: Message, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_source)
    await message.answer("=== NEW LEAD ===\n\nSelect source:", reply_markup=get_source_keyboard())


@router.message(F.text == "Quick")
async def cmd_quick_actions(message: Message, state: FSMContext):
    await message.answer("=== QUICK ===\n\nSelect action:", reply_markup=get_quick_actions_keyboard())


# Media Handlers

@router.message(F.voice)
async def handle_voice(message: Message, state: FSMContext):
    await message.answer(f"Voice: {message.voice.duration}s")

@router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext):
    await message.answer(f"Photo: {message.photo[-1].width}x{message.photo[-1].height}")

@router.message(F.document)
async def handle_document(message: Message, state: FSMContext):
    size_mb = message.document.file_size / (1024 * 1024)
    await message.answer(f"Doc: {message.document.file_name}, {size_mb:.1f}MB")


# Navigation Callbacks

@router.callback_query(F.data == "goto_start")
async def goto_start(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text("=== CRM BOT ===\n\nWelcome! Choose action:", reply_markup=get_start_keyboard())
    await callback.answer()


@router.callback_query(F.data == "goto_menu")
async def goto_menu(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text("=== MAIN MENU ===\n\nChoose action:", reply_markup=get_menu_keyboard())
    await callback.answer()


@router.callback_query(F.data == "goto_settings")
async def goto_settings(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("=== SETTINGS ===\n\nAI: gpt-4o-mini\nMin score: 0.6", reply_markup=get_settings_keyboard())
    await callback.answer()


@router.callback_query(F.data == "goto_leads")
async def goto_leads(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text("=== LEADS ===\n\nSelect section:", reply_markup=get_leads_category_keyboard())
    await callback.answer()


@router.callback_query(F.data == "goto_newlead")
async def goto_newlead(callback: CallbackQuery, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_source)
    await callback.message.edit_text("=== NEW LEAD ===\n\nSelect source:", reply_markup=get_source_keyboard())
    await callback.answer()


@router.callback_query(F.data == "goto_stats")
async def goto_stats(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api()
    total = len(leads)
    new_count = sum(1 for l in leads if l.get("stage") == "new")
    contacted = sum(1 for l in leads if l.get("stage") == "contacted")
    qualified = sum(1 for l in leads if l.get("stage") == "qualified")
    transferred = sum(1 for l in leads if l.get("stage") == "transferred")
    lost = sum(1 for l in leads if l.get("stage") == "lost")
    
    await callback.message.edit_text(
        f"=== STATS ===\n\nTotal: {total}\nNew: {new_count}\nContacted: {contacted}\nQualified: {qualified}\nTransferred: {transferred}\nLost: {lost}\n\nConversion: {round(transferred/total*100, 1) if total > 0 else 0}%"
    )
    await callback.answer()


@router.callback_query(F.data == "goto_quick")
async def goto_quick(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("=== QUICK ===\n\nSelect action:", reply_markup=get_quick_actions_keyboard())
    await callback.answer()


# Subcategory Handlers

@router.callback_query(F.data == "cat_stage")
async def cat_stage(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("=== BY STAGE ===\n\nSelect stage:", reply_markup=get_stage_subcategories_keyboard())
    await callback.answer()


@router.callback_query(F.data == "cat_source")
async def cat_source(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("=== BY SOURCE ===\n\nSelect source:", reply_markup=get_source_subcategories_keyboard())
    await callback.answer()


@router.callback_query(F.data == "cat_domain")
async def cat_domain(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("=== BY DOMAIN ===\n\nSelect domain:", reply_markup=get_domain_subcategories_keyboard())
    await callback.answer()


# Filter Callbacks

@router.callback_query(F.data == "filter_new")
async def filter_new(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(stage="new")
    await show_leads_list(callback, leads, "New Leads")

@router.callback_query(F.data == "filter_contacted")
async def filter_contacted(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(stage="contacted")
    await show_leads_list(callback, leads, "Contacted Leads")

@router.callback_query(F.data == "filter_qualified")
async def filter_qualified(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(stage="qualified")
    await show_leads_list(callback, leads, "Qualified Leads")

@router.callback_query(F.data == "filter_transferred")
async def filter_transferred(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(stage="transferred")
    await show_leads_list(callback, leads, "Transferred Leads")

@router.callback_query(F.data == "filter_lost")
async def filter_lost(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(stage="lost")
    await show_leads_list(callback, leads, "Lost Leads")

@router.callback_query(F.data == "filter_scanner")
async def filter_scanner(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(source="scanner")
    await show_leads_list(callback, leads, "Scanner Leads")

@router.callback_query(F.data == "filter_partner")
async def filter_partner(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(source="partner")
    await show_leads_list(callback, leads, "Partner Leads")

@router.callback_query(F.data == "filter_manual")
async def filter_manual(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(source="manual")
    await show_leads_list(callback, leads, "Manual Leads")

@router.callback_query(F.data == "filter_first")
async def filter_first(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(domain="first")
    await show_leads_list(callback, leads, "First Domain Leads")

@router.callback_query(F.data == "filter_second")
async def filter_second(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(domain="second")
    await show_leads_list(callback, leads, "Second Domain Leads")

@router.callback_query(F.data == "filter_third")
async def filter_third(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api(domain="third")
    await show_leads_list(callback, leads, "Third Domain Leads")

@router.callback_query(F.data == "filter_all")
async def filter_all(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api()
    await show_leads_list(callback, leads, "All Leads")


@router.callback_query(F.data == "filter_myleads")
async def filter_myleads(callback: CallbackQuery, state: FSMContext):
    """Show leads where telegram_id matches current user."""
    user_id = callback.from_user.id
    leads = await get_leads_via_api()
    my_leads = [l for l in leads if str(l.get("telegram_id")) == str(user_id)]
    await show_leads_list(callback, my_leads, "My Leads")


async def show_leads_list(callback: CallbackQuery, leads: list, title: str):
    if not leads:
        await callback.message.edit_text(f"=== {title.upper()} ===\n\nNo leads found.", reply_markup=get_leads_category_keyboard())
    else:
        lead_text = f"=== {title.upper()} ===\n\n"
        for lead in leads:
            stage = lead.get("stage", "?")
            source = lead.get("source", "N/A")
            domain = lead.get("business_domain", "-")
            lead_text += f"#{lead['id']} | {stage} | {source} | {domain}\n"
        
        await callback.message.edit_text(lead_text, reply_markup=get_leads_category_keyboard())
    await callback.answer()


# ===== LEAD DETAIL VIEW =====

@router.callback_query(F.data.startswith("lvw"))
async def handle_lead_view(callback: CallbackQuery, state: FSMContext):
    lead_id = callback.data[4:]
    
    lead = await get_lead_by_id_via_api(int(lead_id))
    
    if lead:
        text = f"=== LEAD #{lead_id} ===\n\n"
        text += f"ID: {lead['id']}\n"
        text += f"Source: {lead.get('source')}\n"
        text += f"Stage: {lead.get('stage')}\n"
        text += f"Domain: {lead.get('business_domain', '-')}\n"
        text += f"Telegram: {lead.get('telegram_id', '-')}\n"
        text += f"Assigned: {lead.get('assigned_to_id', '-')}\n"
        text += f"Messages: {lead.get('message_count', 0)}\n"
        text += f"Score: {lead.get('ai_score', '-')}\n"
        text += f"AI Rec: {lead.get('ai_recommendation', '-')}\n"
        text += f"Created: {lead.get('created_at', '-')[:10]}\n"
        
        await callback.message.edit_text(text, reply_markup=get_lead_detail_keyboard(lead_id))
    else:
        await callback.message.edit_text(f"Lead #{lead_id} not found.", reply_markup=get_menu_keyboard())
    
    await callback.answer()


# Edit Handlers

@router.callback_query(F.data.startswith("led"))
async def handle_lead_edit(callback: CallbackQuery, state: FSMContext):
    data = callback.data[3:]
    parts = data.split("_")
    lead_id = parts[0]
    edit_type = parts[1] if len(parts) > 1 else ""
    
    if edit_type == "stage":
        await callback.message.edit_text(f"=== EDIT STAGE ===\n\nLead #{lead_id}", reply_markup=get_edit_stage_keyboard(lead_id))
    elif edit_type == "del":
        await callback.message.edit_text(f"=== DELETE LEAD ===\n\nAre you sure you want to delete lead #{lead_id}?", reply_markup=get_confirm_delete_keyboard(lead_id))
    elif edit_type == "note":
        await callback.message.edit_text(f"=== ADD NOTE ===\n\nLead #{lead_id}\n\nFeature coming soon!", reply_markup=get_lead_detail_keyboard(lead_id))
    
    await callback.answer()


@router.callback_query(F.data.startswith("eds"))
async def handle_edit_stage(callback: CallbackQuery, state: FSMContext):
    data = callback.data[3:]
    parts = data.split("_")
    lead_id = parts[0]
    new_stage = parts[1] if len(parts) > 1 else ""
    
    result = await update_lead_stage_via_api(int(lead_id), new_stage)
    
    if result:
        await callback.message.edit_text(f"Lead #{lead_id} -> {new_stage}", reply_markup=get_lead_detail_keyboard(lead_id))
    else:
        await callback.message.edit_text(f"Failed to update lead #{lead_id}", reply_markup=get_lead_detail_keyboard(lead_id))
    
    await callback.answer()


@router.callback_query(F.data.startswith("edsrc"))
async def handle_edit_source(callback: CallbackQuery, state: FSMContext):
    data = callback.data[5:]
    parts = data.split("_")
    lead_id = parts[0]
    new_source = parts[1] if len(parts) > 1 else ""
    
    result = await update_lead_via_api(int(lead_id), {"source": new_source})
    
    if result:
        await callback.message.edit_text(f"Lead #{lead_id} source -> {new_source}", reply_markup=get_lead_detail_keyboard(lead_id))
    else:
        await callback.message.edit_text(f"Failed to update lead #{lead_id}", reply_markup=get_lead_detail_keyboard(lead_id))
    
    await callback.answer()


@router.callback_query(F.data.startswith("eddom"))
async def handle_edit_domain(callback: CallbackQuery, state: FSMContext):
    data = callback.data[5:]
    parts = data.split("_")
    lead_id = parts[0]
    new_domain = parts[1] if len(parts) > 1 else ""
    
    domain_value = None if new_domain == "none" else new_domain
    result = await update_lead_via_api(int(lead_id), {"business_domain": domain_value})
    
    if result:
        domain_text = domain_value if domain_value else "removed"
        await callback.message.edit_text(f"Lead #{lead_id} domain -> {domain_text}", reply_markup=get_lead_detail_keyboard(lead_id))
    else:
        await callback.message.edit_text(f"Failed to update lead #{lead_id}", reply_markup=get_lead_detail_keyboard(lead_id))
    
    await callback.answer()


@router.callback_query(F.data.startswith("cfddel"))
async def handle_confirm_delete(callback: CallbackQuery, state: FSMContext):
    data = callback.data[7:]
    parts = data.split("_")
    lead_id = parts[0]
    confirm = parts[1] if len(parts) > 1 else ""
    
    if confirm == "y":
        success = await delete_lead_via_api(int(lead_id))
        if success:
            await callback.message.edit_text(f"Lead #{lead_id} deleted!", reply_markup=get_leads_category_keyboard())
        else:
            await callback.message.edit_text(f"Failed to delete lead #{lead_id}", reply_markup=get_lead_detail_keyboard(lead_id))
    else:
        await callback.message.edit_text("Delete cancelled.", reply_markup=get_lead_detail_keyboard(lead_id))
    
    await callback.answer()


# Lead Actions

@router.callback_query(F.data.startswith("lac"))
async def handle_lead_action(callback: CallbackQuery, state: FSMContext):
    data = callback.data[3:]
    parts = data.split("_")
    lead_id = parts[0]
    action = parts[1] if len(parts) > 1 else ""
    
    stage_map = {"c": "contacted", "q": "qualified", "t": "transferred", "l": "lost"}
    new_stage = stage_map.get(action)
    
    if new_stage and action in ["c", "q", "t", "l"]:
        result = await update_lead_stage_via_api(int(lead_id), new_stage)
        
        if result:
            await callback.message.edit_text(f"Lead #{lead_id} -> {new_stage}", reply_markup=get_lead_detail_keyboard(lead_id))
        else:
            await callback.message.edit_text(f"Failed update #{lead_id}", reply_markup=get_lead_detail_keyboard(lead_id))
    elif action == "a":
        await callback.message.edit_text(f"AI analysis for #{lead_id}...", reply_markup=get_lead_detail_keyboard(lead_id))
    
    await callback.answer()


# Quick Actions

@router.callback_query(F.data == "quick_dashboard")
async def quick_dashboard(callback: CallbackQuery, state: FSMContext):
    leads = await get_leads_via_api()
    total = len(leads)
    transferred = sum(1 for l in leads if l.get("stage") == "transferred")
    
    await callback.message.edit_text(
        f"=== DASHBOARD ===\n\nTotal: {total}\nTransferred: {transferred}\nConversion: {round(transferred/total*100, 1) if total > 0 else 0}%",
        reply_markup=get_menu_keyboard()
    )
    await callback.answer()


@router.callback_query(F.data == "quick_myleads")
async def quick_myleads(callback: CallbackQuery, state: FSMContext):
    """Show leads assigned to current user (by telegram_id)."""
    user_id = callback.from_user.id
    
    # Get all leads and filter by telegram_id
    leads = await get_leads_via_api()
    my_leads = [l for l in leads if str(l.get("telegram_id")) == str(user_id)]
    
    if not my_leads:
        await callback.message.edit_text("=== MY LEADS ===\n\nNo leads assigned to you.", reply_markup=get_menu_keyboard())
    else:
        text = "=== MY LEADS ===\n\n"
        for lead in my_leads:
            text += f"#{lead['id']} | {lead.get('stage')} | {lead.get('source')}\n"
        
        await callback.message.edit_text(text, reply_markup=get_menu_keyboard())
    await callback.answer()


@router.callback_query(F.data == "quick_addlead")
async def quick_addlead(callback: CallbackQuery, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_source)
    await callback.message.edit_text("=== NEW LEAD ===\n\nSelect source:", reply_markup=get_source_keyboard())
    await callback.answer()


@router.callback_query(F.data == "quick_refresh")
async def quick_refresh(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Refreshed!", reply_markup=get_menu_keyboard())
    await callback.answer()


# Source and Domain for new lead

@router.callback_query(F.data.startswith("src_"))
async def handle_source(callback: CallbackQuery, state: FSMContext):
    source = callback.data.replace("src_", "")
    await state.update_data(source=source)
    await state.set_state(LeadCreationState.waiting_for_domain)
    
    await callback.message.edit_text(f"Source: {source.upper()}\n\nSelect domain:", reply_markup=get_domain_keyboard())
    await callback.answer()


@router.callback_query(F.data.startswith("dmn_"))
async def handle_domain(callback: CallbackQuery, state: FSMContext):
    domain = callback.data.replace("dmn_", "")
    if domain == "skip":
        domain = None
    if domain:
        await state.update_data(business_domain=domain)
    
    data = await state.get_data()
    source = data.get("source", "manual")
    telegram_id = callback.from_user.id
    
    lead = await create_lead_via_api(source=source, domain=domain, telegram_id=telegram_id)
    
    if lead:
        await callback.message.edit_text(f"=== CREATED ===\n\nID: {lead['id']}\nSource: {lead['source']}\nStage: {lead['stage']}", reply_markup=get_menu_keyboard())
    else:
        await callback.message.edit_text("Failed to create lead.", reply_markup=get_menu_keyboard())
    
    await state.clear()
    await callback.answer()


# Error Handlers

@router.errors()
async def handle_errors(event: Exception):
    logger.error(f"Bot error: {event}")


def get_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.include_router(router)
    return dp


async def start_bot():
    bot = get_bot()
    dp = get_dispatcher()
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(start_bot())
