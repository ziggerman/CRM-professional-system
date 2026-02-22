"""
Telegram Bot handlers — Professional UI/UX with rich HTML formatting.
Uses ui.py for all message formatting and keyboards.py for all keyboards.
"""
import logging
import os
from io import BytesIO
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
    get_lead_list_keyboard,
    get_note_cancel_keyboard,
    get_search_cancel_keyboard,
    get_back_keyboard,
    get_back_to_menu_keyboard,
    get_retry_keyboard,
    # Sales Keyboards
    get_sales_category_keyboard,
    get_sale_stage_categories_keyboard,
    get_sales_list_keyboard,
    get_sale_detail_keyboard,
    get_edit_sale_stage_keyboard,
)
from app.bot.states import LeadCreationState, AddNoteState, SearchState, SaleManagementState
from app.bot import ui

logger = logging.getLogger(__name__)

router = Router()
bot: Optional[Bot] = None

LEADS_PAGE_SIZE = 7  # Leads per page in list view


# ─────────────────────────────────────────────────────────────
# Bot Instance
# ─────────────────────────────────────────────────────────────

def get_bot() -> Bot:
    global bot
    if bot is None:
        bot = Bot(token=bot_settings.TELEGRAM_BOT_TOKEN)
    return bot


# ─────────────────────────────────────────────────────────────
# API Client Functions
# ─────────────────────────────────────────────────────────────

async def _get_role_header(telegram_id: str | int = None) -> dict:
    """Fetch user role and build X-User-Role header."""
    if not telegram_id:
        return {}
    
    # Priority 1: Check if admin by static ID list (Step 3.1 fallback)
    from app.bot.config import bot_settings
    if int(telegram_id) in bot_settings.TELEGRAM_ADMIN_IDS:
        return {"X-User-Role": "ADMIN"}

    import httpx
    url = f"http://localhost:8000/api/v1/users/me?telegram_id={telegram_id}"
    auth_header = {"Authorization": f"Bearer {bot_settings.API_SECRET_TOKEN}"} if hasattr(bot_settings, 'API_SECRET_TOKEN') else {}
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=auth_header, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return {"X-User-Role": data.get("role", "agent").upper()}
    except Exception as e:
        logger.error(f"Failed to fetch role for {telegram_id}: {e}")
        
    return {"X-User-Role": "AGENT"}


async def _upload_file_to_api(lead_id: int, file_id: str, file_name: str, user_id: int = None) -> Optional[dict]:
    """Download file from Telegram and upload to Lead API."""
    import httpx
    bot_instance = get_bot()
    
    try:
        file = await bot_instance.get_file(file_id)
        file_content = await bot_instance.download_file(file.file_path)
        
        url = f"http://localhost:8000/api/v1/leads/{lead_id}/attachments"
        headers = {"Authorization": f"Bearer {bot_settings.API_SECRET_TOKEN}"} if hasattr(bot_settings, 'API_SECRET_TOKEN') else {}
        if user_id:
            headers.update(await _get_role_header(user_id))
            
        files = {"file": (file_name, file_content)}
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, files=files, timeout=30.0)
            if resp.status_code in (200, 201):
                return resp.json()
    except Exception as e:
        logger.error(f"File upload failed for lead {lead_id}: {e}")
        
    return None


async def _api_get(path: str, user_id: int = None) -> Optional[dict]:
    import httpx
    url = f"http://localhost:8000{path}"
    headers = {"Authorization": f"Bearer {bot_settings.API_SECRET_TOKEN}"} if hasattr(bot_settings, 'API_SECRET_TOKEN') else {}
    if user_id:
        headers.update(await _get_role_header(user_id))
        
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return {"error": True, "detail": response.json().get("detail", "Unknown error"), "status": response.status_code}
        except Exception as e:
            logger.error(f"API GET {path} error: {e}")
    return {"error": True, "detail": "Connection error"}


async def _api_post(path: str, data: dict, user_id: int = None) -> Optional[dict]:
    import httpx
    url = f"http://localhost:8000{path}"
    headers = {"Authorization": f"Bearer {bot_settings.API_SECRET_TOKEN}"} if hasattr(bot_settings, 'API_SECRET_TOKEN') else {}
    if user_id:
        headers.update(await _get_role_header(user_id))
        
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=data, headers=headers, timeout=10.0)
            if response.status_code in (200, 201):
                return response.json()
            return {"error": True, "detail": response.json().get("detail", "Unknown error"), "status": response.status_code}
        except Exception as e:
            logger.error(f"API POST {path} error: {e}")
    return {"error": True, "detail": "Connection error"}


def is_valid_email(email: str) -> bool:
    import re
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))


def is_valid_phone(phone: str) -> bool:
    import re
    # Basic check for + and digits, min 7 chars
    return bool(re.match(r"^\+?[0-9\s\-]{7,20}$", phone))


async def _api_patch(path: str, data: dict, user_id: int = None) -> Optional[dict]:
    import httpx
    url = f"http://localhost:8000{path}"
    headers = {"Authorization": f"Bearer {bot_settings.API_SECRET_TOKEN}"} if hasattr(bot_settings, 'API_SECRET_TOKEN') else {}
    if user_id:
        headers.update(await _get_role_header(user_id))
        
    async with httpx.AsyncClient() as client:
        try:
            response = await client.patch(url, json=data, headers=headers, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return {"error": True, "detail": response.json().get("detail", "Unknown error"), "status": response.status_code}
        except Exception as e:
            logger.error(f"API PATCH {path} error: {e}")
    return {"error": True, "detail": "Connection error"}


async def _api_delete(path: str, user_id: int = None) -> bool:
    import httpx
    url = f"http://localhost:8000{path}"
    headers = {"Authorization": f"Bearer {bot_settings.API_SECRET_TOKEN}"} if hasattr(bot_settings, 'API_SECRET_TOKEN') else {}
    if user_id:
        headers.update(await _get_role_header(user_id))
        
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(url, headers=headers, timeout=10.0)
            return response.status_code in (200, 204)
        except Exception as e:
            logger.error(f"API DELETE {path} error: {e}")
    return False


async def create_lead_via_api(source: str, domain: str = None, telegram_id: int = None, user_id: int = None) -> Optional[dict]:
    data = {"source": source, "telegram_id": str(telegram_id) if telegram_id else None}
    if domain:
        data["business_domain"] = domain
    return await _api_post("/api/v1/leads", data, user_id=user_id)


async def get_leads_via_api(
    stage: str = None, 
    source: str = None, 
    domain: str = None, 
    assigned_to_id: int = None, 
    user_id: int = None,
    query: str = None,
    created_after: str = None,
    created_before: str = None
) -> list:
    params = []
    if stage:
        params.append(f"stage={stage}")
    if source:
        params.append(f"source={source}")
    if domain:
        params.append(f"business_domain={domain}")
    if assigned_to_id:
        params.append(f"assigned_to_id={assigned_to_id}")
    if query:
        import urllib.parse
        params.append(f"query={urllib.parse.quote(query)}")
    if created_after:
        params.append(f"created_after={created_after}")
    if created_before:
        params.append(f"created_before={created_before}")

    query = "?" + "&".join(params) if params else ""
    data = await _api_get(f"/api/v1/leads{query}", user_id=user_id)
    if data and isinstance(data, dict):
        return data.get("items", [])
    return []


async def get_lead_by_id_via_api(lead_id: int, user_id: int = None) -> Optional[dict]:
    return await _api_get(f"/api/v1/leads/{lead_id}", user_id=user_id)


async def update_lead_via_api(lead_id: int, data: dict, user_id: int = None) -> Optional[dict]:
    return await _api_patch(f"/api/v1/leads/{lead_id}", data, user_id=user_id)


async def update_lead_stage_via_api(lead_id: int, stage: str, user_id: int = None) -> Optional[dict]:
    return await _api_patch(f"/api/v1/leads/{lead_id}/stage", {"stage": stage}, user_id=user_id)


async def delete_lead_via_api(lead_id: int, user_id: int = None) -> bool:
    return await _api_delete(f"/api/v1/leads/{lead_id}", user_id)


async def get_dashboard_via_api(user_id: int = None) -> Optional[dict]:
    return await _api_get("/api/v1/dashboard", user_id=user_id)


# ─────────────────────────────────────────────────────────────
# UX Helpers
# ─────────────────────────────────────────────────────────────

async def safe_edit(callback: CallbackQuery, text: str, markup=None, parse_mode: str = "HTML"):
    """Edit message safely — answer callback regardless of errors."""
    try:
        await callback.message.edit_text(text, reply_markup=markup, parse_mode=parse_mode)
    except Exception as e:
        logger.warning(f"safe_edit failed: {e}")
    finally:
        await callback.answer()


async def show_lead_detail(callback: CallbackQuery, lead_id: int):
    """Render and show the lead detail view."""
    lead = await get_lead_by_id_via_api(lead_id, user_id=callback.from_user.id)
    if lead:
        text = ui.format_lead_card(lead)
        stage = lead.get("stage")
        await safe_edit(callback, text, get_lead_detail_keyboard(lead_id, stage))
    else:
        await safe_edit(
            callback,
            ui.format_error(f"Lead #{lead_id} not found.", "ID may be invalid or deleted."),
            get_back_to_menu_keyboard()
        )


async def show_leads_list_page(callback: CallbackQuery, leads: list, title: str, page: int = 0, back_cb: str = "goto_leads"):
    """Show a paginated leads list."""
    total_leads = len(leads)
    total_pages = max(1, (total_leads + LEADS_PAGE_SIZE - 1) // LEADS_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    page_leads = leads[page * LEADS_PAGE_SIZE:(page + 1) * LEADS_PAGE_SIZE]

    header = ui.format_leads_list(leads, title, page, total_pages)
    keyboard = get_lead_list_keyboard(page_leads, page, total_pages, back_cb)
    await safe_edit(callback, header, keyboard)


# ─────────────────────────────────────────────────────────────
# Command Handlers
# ─────────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    user = message.from_user
    is_admin = user.id in bot_settings.TELEGRAM_ADMIN_IDS

    await message.answer(
        ui.format_welcome(user.first_name, is_admin),
        reply_markup=get_main_menu_keyboard(),
        parse_mode="HTML"
    )
    await message.answer(
        "📋 <b>Choose where to start:</b>",
        reply_markup=get_start_keyboard(),
        parse_mode="HTML"
    )


@router.message(Command("menu"))
@router.message(F.text.in_(["Menu", "🏠 Menu"]))
async def cmd_menu(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "📋 <b>MAIN MENU</b>\n\nChoose an option:",
        reply_markup=get_menu_keyboard(),
        parse_mode="HTML"
    )


@router.message(Command("help"))
async def cmd_help(message: Message, state: FSMContext):
    await message.answer(ui.format_help(), reply_markup=get_back_to_menu_keyboard(), parse_mode="HTML")


@router.message(Command("settings"))
@router.message(F.text == "⚙️ Settings")
async def cmd_settings(message: Message, state: FSMContext):
    await message.answer(
        ui.format_settings(),
        reply_markup=get_settings_keyboard(),
        parse_mode="HTML"
    )


@router.message(F.text == "📋 Leads")
async def cmd_leads(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "📋 <b>LEADS</b>\n\nSelect a view:",
        reply_markup=get_leads_category_keyboard(),
        parse_mode="HTML"
    )


@router.message(F.text == "💰 Sales")
async def cmd_sales(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "💰 <b>SALES PIPELINE</b>\n\nSelect a view:",
        reply_markup=get_sales_category_keyboard(),
        parse_mode="HTML"
    )


@router.message(F.text == "📊 Stats")
async def cmd_statistics(message: Message, state: FSMContext):
    leads = await get_leads_via_api(user_id=message.from_user.id)
    await message.answer(
        ui.format_stats_simple(leads),
        parse_mode="HTML"
    )


@router.message(F.text == "➕ New Lead")
async def cmd_new_lead(message: Message, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_source)
    await message.answer(
        "➕ <b>NEW LEAD</b>  <i>Step 1 of 2</i>\n\nSelect the lead source:",
        reply_markup=get_source_keyboard(),
        parse_mode="HTML"
    )


@router.message(F.text == "⚡ Quick")
async def cmd_quick_actions(message: Message, state: FSMContext):
    await message.answer(
        "⚡ <b>QUICK ACTIONS</b>\n\nChoose an action:",
        reply_markup=get_quick_actions_keyboard(),
        parse_mode="HTML"
    )


# ─────────────────────────────────────────────────────────────
# Media Handlers
# ─────────────────────────────────────────────────────────────

@router.message(F.voice)
async def handle_voice(message: Message, state: FSMContext):
    await message.answer(
        f"🎤 <b>Voice Received</b>\n\nDuration: <b>{message.voice.duration}s</b>\n\n"
        f"<i>Voice processing is not yet configured.</i>",
        parse_mode="HTML"
    )


@router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext):
    # If in AddNoteState, handle as attachment
    curr_state = await state.get_state()
    if curr_state == AddNoteState.waiting_for_text:
        data = await state.get_data()
        lead_id = data.get("note_lead_id")
        if lead_id:
            photo = message.photo[-1]
            await message.answer("📤 <i>Uploading photo to lead...</i>", parse_mode="HTML")
            res = await _upload_file_to_api(lead_id, photo.file_id, f"photo_{photo.file_id[:8]}.jpg", message.from_user.id)
            if res:
                await message.answer(ui.format_success(f"Photo attached to Lead #{lead_id}."), parse_mode="HTML")
            else:
                await message.answer(ui.format_error("Failed to upload photo."), parse_mode="HTML")
            await state.clear()
            return

    photo = message.photo[-1]
    await message.answer(
        f"📷 <b>Photo Received</b>\n\nResolution: <b>{photo.width}×{photo.height}</b>\n\n"
        f"<i>To attach this to a lead, use the <b>'Add Note'</b> feature on the lead's card.</i>",
        parse_mode="HTML"
    )


@router.message(F.document)
async def handle_document(message: Message, state: FSMContext):
    # If in AddNoteState, handle as attachment
    curr_state = await state.get_state()
    if curr_state == AddNoteState.waiting_for_text:
        data = await state.get_data()
        lead_id = data.get("note_lead_id")
        if lead_id:
            await message.answer(f"📤 <i>Uploading <b>{message.document.file_name}</b>...</i>", parse_mode="HTML")
            res = await _upload_file_to_api(lead_id, message.document.file_id, message.document.file_name, message.from_user.id)
            if res:
                await message.answer(ui.format_success(f"File attached to Lead #{lead_id}."), parse_mode="HTML")
            else:
                await message.answer(ui.format_error("Failed to upload file."), parse_mode="HTML")
            await state.clear()
            return

    size_mb = message.document.file_size / (1024 * 1024)
    await message.answer(
        f"📄 <b>Document Received</b>\n\n"
        f"File: <code>{message.document.file_name}</code>\n"
        f"Size: <b>{size_mb:.2f} MB</b>\n\n"
        f"<i>To attach this to a lead, use the <b>'Add Note'</b> feature on the lead's card.</i>",
        parse_mode="HTML"
    )


# ─────────────────────────────────────────────────────────────
# Navigation Callbacks
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data == "goto_start")
async def goto_start(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    user = callback.from_user
    is_admin = user.id in bot_settings.TELEGRAM_ADMIN_IDS
    await safe_edit(callback, ui.format_welcome(user.first_name, is_admin), get_start_keyboard())


@router.callback_query(F.data == "goto_menu")
async def goto_menu(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await safe_edit(callback, "📋 <b>MAIN MENU</b>\n\nChoose an option:", get_menu_keyboard())


@router.callback_query(F.data == "goto_settings")
async def goto_settings(callback: CallbackQuery, state: FSMContext):
    await safe_edit(callback, ui.format_settings(), get_settings_keyboard())


@router.callback_query(F.data == "goto_leads")
async def goto_leads(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await safe_edit(callback, "📋 <b>LEADS</b>\n\nSelect a view:", get_leads_category_keyboard())


@router.callback_query(F.data == "goto_sales")
async def goto_sales(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await safe_edit(callback, "💰 <b>SALES PIPELINE</b>\n\nSelect a view:", get_sales_category_keyboard())


@router.callback_query(F.data == "scat_stage")
async def scat_stage(callback: CallbackQuery, state: FSMContext):
    await safe_edit(callback, "📈 <b>SALES BY STAGE</b>\n\nSelect stage:", get_sale_stage_categories_keyboard())


async def _api_get_sales(user_id: int, stage: str = None) -> list:
    """Helper to fetch sales from API."""
    url = "/api/v1/sales"
    if stage:
        url += f"?stage={stage}"
    res = await _api_get(url, user_id=user_id)
    return res.get("items", []) if res else []


@router.callback_query(F.data.startswith("filter_sales_all"))
@router.callback_query(F.data.startswith("filter_mysales"))
@router.callback_query(F.data.startswith("sfilter_"))
@router.callback_query(F.data.startswith("spg"))
async def handle_sales_filtering(callback: CallbackQuery, state: FSMContext):
    page = 0
    if callback.data.startswith("spg"):
        page = int(callback.data[3:])
        data = await state.get_data()
        stage = data.get("sale_filter_stage")
        title = data.get("sale_filter_title", "All Sales")
    else:
        stage = None
        title = "All Sales"
        if callback.data.startswith("sfilter_"):
            stage = callback.data.replace("sfilter_", "")
            title = f"Sales: {stage.title()}"
        elif callback.data == "filter_mysales":
            # API doesn't have explicit 'my sales' yet, but we'll filter by assigned lead
            # For now, just show all as placeholder or add filter if API supports it
            title = "My Sales"
            
        await state.update_data(sale_filter_stage=stage, sale_filter_title=title)

    sales = await _api_get_sales(user_id=callback.from_user.id, stage=stage)
    
    # Simple pagination
    page_size = 5
    total_pages = (len(sales) + page_size - 1) // page_size
    start = page * page_size
    subset = sales[start:start + page_size]

    await safe_edit(
        callback, 
        ui.format_sales_list(subset, title, page, total_pages),
        get_sales_list_keyboard(subset, page, total_pages)
    )


@router.callback_query(F.data.startswith("svw"))
async def handle_sale_view(callback: CallbackQuery, state: FSMContext):
    sale_id = int(callback.data[3:])
    await show_sale_detail(callback, sale_id)


async def show_sale_detail(callback: CallbackQuery, sale_id: int):
    sale = await _api_get(f"/api/v1/sales/{sale_id}", user_id=callback.from_user.id)
    if not sale or "error" in sale:
        await callback.answer("Sale not found", show_alert=True)
        return
        
    await safe_edit(
        callback,
        ui.format_sale_card(sale),
        get_sale_detail_keyboard(sale_id, sale.get("stage"))
    )


@router.callback_query(F.data.regexp(r"^sed\d"))
async def handle_sale_action(callback: CallbackQuery, state: FSMContext):
    raw = callback.data[3:]
    parts = raw.split("_", 1)
    sale_id = int(parts[0])
    action = parts[1]

    if action == "stage":
        sale = await _api_get(f"/api/v1/sales/{sale_id}", user_id=callback.from_user.id)
        await safe_edit(callback, "📈 <b>UPDATE STAGE</b>\n\nSelect next step:", get_edit_sale_stage_keyboard(sale_id, sale.get("stage")))
    elif action == "lview":
        sale = await _api_get(f"/api/v1/sales/{sale_id}", user_id=callback.from_user.id)
        if sale and "error" not in sale and "lead_id" in sale:
            await show_lead_detail(callback, sale["lead_id"])
        else:
            await callback.answer("Record not found", show_alert=True)
    elif action == "amt":
        await state.update_data(edit_sale_id=sale_id)
        await state.set_state(SaleManagementState.updating_notes) # Re-using or extending states
        # I'll use a new state for clarity if needed, but SaleManagementState.updating_notes is generic enough or I'll add a new one.
        # Let's check states.py again. (Id 1848)
        # class SaleManagementState(StatesGroup):
        #     updating_notes = State()
        #     confirming_action = State()
        
        # Actually SaleManagementState.updating_notes is fine for both notes/amount if we handle by context.
        await callback.answer("Editing amount...")
        await callback.message.answer("💰 <b>Enter new amount</b> (in USD cents, e.g. 5000 for $50.00):", parse_mode="HTML")
    elif action == "nt":
        await state.update_data(edit_sale_id=sale_id)
        await state.set_state(SaleManagementState.updating_notes)
        await callback.answer("Editing notes...")
        await callback.message.answer("📝 <b>Enter sale notes:</b>", parse_mode="HTML")


@router.callback_query(F.data.startswith("seds"))
async def handle_sale_stage_change(callback: CallbackQuery, state: FSMContext):
    raw = callback.data[4:]
    parts = raw.split("_", 1)
    sale_id = int(parts[0])
    new_stage = parts[1]
    
    await callback.answer("Updating pipeline...")
    url = f"/api/v1/sales/{sale_id}/stage"
    res = await _api_patch(url, {"stage": new_stage}, user_id=callback.from_user.id)
    
    if res and "error" not in res:
        await callback.answer(f"✅ Sale #{sale_id} updated to {new_stage}!", show_alert=True)
        await show_sale_detail(callback, sale_id)
    else:
        await callback.answer(f"⚠️ Failed: {res.get('detail', 'Unknown error')}", show_alert=True)


@router.message(SaleManagementState.updating_notes)
async def handle_sale_input(message: Message, state: FSMContext):
    data = await state.get_data()
    sale_id = data.get("edit_sale_id")
    if not sale_id:
        await state.clear()
        return

    text = message.text or ""
    payload = {}
    
    # Try to parse as amount if it's purely digits
    if text.isdigit():
        payload["amount"] = int(text)
    else:
        payload["notes"] = text

    await _api_patch(f"/api/v1/sales/{sale_id}", payload, user_id=message.from_user.id)
    await state.clear()
    await message.answer(f"✅ Sale #{sale_id} updated.", parse_mode="HTML")


@router.callback_query(F.data == "goto_newlead")
async def goto_newlead(callback: CallbackQuery, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_source)
    await safe_edit(
        callback,
        ui.format_lead_creation_step("Start", "📡 SELECT SOURCE", "Where did this lead come from?"),
        get_source_keyboard()
    )


@router.callback_query(F.data == "back_name", LeadCreationState.waiting_for_email)
async def back_name(callback: CallbackQuery, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_name)
    await safe_edit(callback, ui.format_lead_creation_step("1/8", "👤 FULL NAME", "Please enter the lead's full name:"), get_name_keyboard())

@router.callback_query(F.data == "back_email", LeadCreationState.waiting_for_phone)
async def back_email(callback: CallbackQuery, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_email)
    await safe_edit(callback, ui.format_lead_creation_step("2/8", "📧 EMAIL ADDRESS", "Enter a valid email address:"), get_email_keyboard())

@router.callback_query(F.data == "back_phone", LeadCreationState.waiting_for_username)
async def back_phone(callback: CallbackQuery, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_phone)
    await safe_edit(callback, ui.format_lead_creation_step("3/8", "📞 PHONE NUMBER", "Enter phone number (e.g. +1234567890):"), get_phone_keyboard())

@router.callback_query(F.data == "back_username", LeadCreationState.waiting_for_domain)
async def back_username(callback: CallbackQuery, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_username)
    await safe_edit(callback, ui.format_lead_creation_step("4/8", "📡 MESSENGER USERNAME", "Enter Telegram/WhatsApp username:"), get_username_keyboard())

@router.callback_query(F.data == "back_domain", LeadCreationState.waiting_for_intent)
async def back_domain(callback: CallbackQuery, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_domain)
    await safe_edit(callback, ui.format_lead_creation_step("5/8", "🌐 BUSINESS DOMAIN", "Select the relevant industry:"), get_domain_keyboard("5/8"))

@router.callback_query(F.data == "back_intent")
async def back_intent(callback: CallbackQuery, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_intent)
    await safe_edit(callback, ui.format_lead_creation_step("6/8", "🎯 INTENT / ACTION", "What action did the lead take?"), get_intent_keyboard())


@router.callback_query(F.data == "goto_stats")
async def goto_stats(callback: CallbackQuery, state: FSMContext):
    await callback.answer("Loading stats...")
    leads = await get_leads_via_api(user_id=callback.from_user.id)
    await safe_edit(callback, ui.format_stats_simple(leads))


@router.callback_query(F.data == "goto_quick")
async def goto_quick(callback: CallbackQuery, state: FSMContext):
    await safe_edit(callback, "⚡ <b>QUICK ACTIONS</b>\n\nChoose an action:", get_quick_actions_keyboard())


@router.callback_query(F.data == "goto_dashboard")
async def goto_dashboard(callback: CallbackQuery, state: FSMContext):
    await callback.answer("Loading dashboard...")
    stats = await get_dashboard_via_api(user_id=callback.from_user.id)
    if stats and "error" not in stats:
        await safe_edit(callback, ui.format_dashboard(stats), get_dashboard_keyboard())
    else:
        # Fallback: show simple stats from leads list
        leads = await get_leads_via_api(user_id=callback.from_user.id)
        await safe_edit(callback, ui.format_stats_simple(leads), get_dashboard_keyboard())


@router.callback_query(F.data == "goto_search")
async def goto_search(callback: CallbackQuery, state: FSMContext):
    await state.set_state(SearchState.waiting_for_query)
    await safe_edit(callback, ui.format_search_prompt(), get_search_cancel_keyboard())


@router.callback_query(F.data == "noop")
async def noop(callback: CallbackQuery):
    """No-op for pagination page indicator button."""
    await callback.answer()


# ─────────────────────────────────────────────────────────────
# Settings Callbacks
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data == "settings_notif")
async def settings_notif(callback: CallbackQuery):
    await safe_edit(
        callback,
        "🔔 <b>NOTIFICATIONS</b>\n\n<i>Notification settings coming soon.</i>",
        get_back_keyboard("goto_settings")
    )


@router.callback_query(F.data == "settings_ai")
async def settings_ai(callback: CallbackQuery):
    await safe_edit(
        callback,
        "🤖 <b>AI SETTINGS</b>\n\n"
        "├─ Model: <code>gpt-4o-mini</code>\n"
        "├─ Min Score: <code>0.60</code>\n"
        "└─ Auto-analyze: <code>OFF</code>\n\n"
        "<i>Advanced AI settings coming soon.</i>",
        get_back_keyboard("goto_settings")
    )


@router.callback_query(F.data == "settings_profile")
async def settings_profile(callback: CallbackQuery):
    user = callback.from_user
    is_admin = user.id in bot_settings.TELEGRAM_ADMIN_IDS
    text = (
        f"👤 <b>MY PROFILE</b>\n\n"
        f"├─ Name: <b>{user.full_name}</b>\n"
        f"├─ Username: @{user.username or '—'}\n"
        f"├─ Telegram ID: <code>{user.id}</code>\n"
        f"└─ Role: {'👑 Admin' if is_admin else '👤 User'}"
    )
    await safe_edit(callback, text, get_back_keyboard("goto_settings"))


# ─────────────────────────────────────────────────────────────
# Lead Category / Filter Subcategories
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data == "cat_stage")
async def cat_stage(callback: CallbackQuery):
    await safe_edit(callback, "📈 <b>BY STAGE</b>\n\nSelect a stage:", get_stage_subcategories_keyboard())


@router.callback_query(F.data == "cat_source")
async def cat_source(callback: CallbackQuery):
    await safe_edit(callback, "📥 <b>BY SOURCE</b>\n\nSelect a source:", get_source_subcategories_keyboard())


@router.callback_query(F.data == "cat_domain")
async def cat_domain(callback: CallbackQuery):
    await safe_edit(callback, "🏢 <b>BY DOMAIN</b>\n\nSelect a domain:", get_domain_subcategories_keyboard())


# ─────────────────────────────────────────────────────────────
# Filter → Lead List
# ─────────────────────────────────────────────────────────────

async def _show_filtered_leads(callback: CallbackQuery, title: str, **kwargs):
    """Generic filter handler: fetch leads, show paginated list."""
    await callback.answer("Loading...")
    leads = await get_leads_via_api(user_id=callback.from_user.id, **kwargs)
    await show_leads_list_page(callback, leads, title)


@router.callback_query(F.data == "filter_new")
async def filter_new(callback: CallbackQuery):
    await _show_filtered_leads(callback, "🆕 New Leads", stage="new")


@router.callback_query(F.data == "filter_contacted")
async def filter_contacted(callback: CallbackQuery):
    await _show_filtered_leads(callback, "📞 Contacted Leads", stage="contacted")


@router.callback_query(F.data == "filter_qualified")
async def filter_qualified(callback: CallbackQuery):
    await _show_filtered_leads(callback, "✅ Qualified Leads", stage="qualified")


@router.callback_query(F.data == "filter_transferred")
async def filter_transferred(callback: CallbackQuery):
    await _show_filtered_leads(callback, "🚀 Transferred Leads", stage="transferred")


@router.callback_query(F.data == "filter_lost")
async def filter_lost(callback: CallbackQuery):
    await _show_filtered_leads(callback, "❌ Lost Leads", stage="lost")


@router.callback_query(F.data == "filter_scanner")
async def filter_scanner(callback: CallbackQuery):
    await _show_filtered_leads(callback, "📡 Scanner Leads", source="scanner")


@router.callback_query(F.data == "filter_partner")
async def filter_partner(callback: CallbackQuery):
    await _show_filtered_leads(callback, "🤝 Partner Leads", source="partner")


@router.callback_query(F.data == "filter_manual")
async def filter_manual(callback: CallbackQuery):
    await _show_filtered_leads(callback, "✏️ Manual Leads", source="manual")


@router.callback_query(F.data == "filter_first")
async def filter_first(callback: CallbackQuery):
    await _show_filtered_leads(callback, "1️⃣ First Domain Leads", domain="first")


@router.callback_query(F.data == "filter_second")
async def filter_second(callback: CallbackQuery):
    await _show_filtered_leads(callback, "2️⃣ Second Domain Leads", domain="second")


@router.callback_query(F.data == "filter_third")
async def filter_third(callback: CallbackQuery):
    await _show_filtered_leads(callback, "3️⃣ Third Domain Leads", domain="third")


@router.callback_query(F.data == "filter_all")
async def filter_all(callback: CallbackQuery):
    await _show_filtered_leads(callback, "📋 All Leads")


@router.callback_query(F.data == "filter_myleads")
async def filter_myleads(callback: CallbackQuery):
    await callback.answer("Loading...")
    user_id = callback.from_user.id
    all_leads = await get_leads_via_api(user_id=user_id)
    my_leads = [l for l in all_leads if str(l.get("telegram_id")) == str(user_id)]
    await show_leads_list_page(callback, my_leads, "👤 My Leads")


# ─────────────────────────────────────────────────────────────
# Pagination
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("pg"))
async def handle_pagination(callback: CallbackQuery, state: FSMContext):
    """Handle pagination — restore the last list from state."""
    page = int(callback.data[2:])
    data = await state.get_data()
    leads_cache = data.get("leads_cache", [])
    title = data.get("leads_title", "Leads")

    if leads_cache:
        await show_leads_list_page(callback, leads_cache, title, page)
    else:
        await safe_edit(callback, "⚠️ Session expired. Please re-open the list.", get_back_to_menu_keyboard())


# ─────────────────────────────────────────────────────────────
# Lead Detail View
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("lvw"))
async def handle_lead_view(callback: CallbackQuery, state: FSMContext):
    lead_id_str = callback.data[3:]
    try:
        lead_id = int(lead_id_str)
    except ValueError:
        await callback.answer("Invalid lead ID", show_alert=True)
        return
    await callback.answer("Opening lead...")
    await show_lead_detail(callback, lead_id)


# ─────────────────────────────────────────────────────────────
# Lead Edit Handlers
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("led"))
async def handle_lead_edit(callback: CallbackQuery, state: FSMContext):
    """Dispatch lead edit actions: stage, source, domain, note, delete."""
    raw = callback.data[3:]
    parts = raw.split("_", 1)
    lead_id_str = parts[0]
    edit_type = parts[1] if len(parts) > 1 else ""

    try:
        lead_id = int(lead_id_str)
    except ValueError:
        await callback.answer("Invalid lead", show_alert=True)
        return

    if edit_type == "stage":
        lead = await get_lead_by_id_via_api(lead_id, user_id=callback.from_user.id)
        current_stage = lead.get("stage") if lead else None
        await safe_edit(
            callback,
            f"✏️ <b>EDIT STAGE</b>  —  Lead #{lead_id}\n\n"
            f"Current: {ui.fmt_stage(current_stage)}\n\n"
            f"Select new stage:",
            get_edit_stage_keyboard(lead_id, current_stage)
        )

    elif edit_type == "src":
        lead = await get_lead_by_id_via_api(lead_id, user_id=callback.from_user.id)
        current_source = lead.get("source") if lead and "error" not in lead else None
        await safe_edit(
            callback,
            f"✏️ <b>EDIT SOURCE</b>  —  Lead #{lead_id}\n\n"
            f"Current: {ui.fmt_source(current_source)}\n\n"
            f"Select new source:",
            get_edit_source_keyboard(lead_id, current_source)
        )

    elif edit_type == "dom":
        lead = await get_lead_by_id_via_api(lead_id, user_id=callback.from_user.id)
        current_domain = lead.get("business_domain") if lead and "error" not in lead else None
        await safe_edit(
            callback,
            f"✏️ <b>EDIT DOMAIN</b>  —  Lead #{lead_id}\n\n"
            f"Current: {ui.fmt_domain(current_domain)}\n\n"
            f"Select new domain:",
            get_edit_domain_keyboard(lead_id, current_domain)
        )

    elif edit_type == "ntm":
        notes = await _api_get(f"/api/v1/leads/{lead_id}/notes", user_id=callback.from_user.id)
        count = notes.get("total", 0) if notes and "error" not in notes else 0
        await safe_edit(
            callback,
            ui.format_notes_menu(lead_id, count),
            get_notes_manage_keyboard(lead_id, has_notes=(count > 0))
        )

    elif edit_type == "ntadd":
        await state.set_state(AddNoteState.waiting_for_text)
        await state.update_data(note_lead_id=lead_id)
        await safe_edit(
            callback,
            ui.format_note_prompt(lead_id),
            get_note_cancel_keyboard(lead_id)
        )

    elif edit_type == "ntvw":
        notes_data = await _api_get(f"/api/v1/leads/{lead_id}/notes", user_id=callback.from_user.id)
        notes = notes_data.get("items", []) if notes_data and "error" not in notes_data else []
        if not notes:
            await callback.answer("No notes found or error loading notes.")
            return

        await state.update_data(viewing_notes=notes, viewing_lead_id=lead_id)
        note = notes[0]
        await safe_edit(
            callback,
            ui.format_single_note(lead_id, note, 0, len(notes)),
            get_note_view_keyboard(lead_id, note['id'], 0, len(notes))
        )

    elif edit_type == "del":
        await safe_edit(
            callback,
            ui.format_delete_confirm(lead_id),
            get_confirm_delete_keyboard(lead_id)
        )


# ─────────────────────────────────────────────────────────────
# Edit Stage — Apply
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("eds"))
async def handle_edit_stage(callback: CallbackQuery, state: FSMContext):
    raw = callback.data[3:]
    parts = raw.split("_", 1)
    lead_id_str = parts[0]
    new_stage = parts[1] if len(parts) > 1 else ""

    try:
        lead_id = int(lead_id_str)
    except ValueError:
        await callback.answer("Invalid lead", show_alert=True)
        return

    await callback.answer("Updating stage...")
    result = await update_lead_stage_via_api(lead_id, new_stage, user_id=callback.from_user.id)
    if result and "error" not in result:
        stage_label = ui.fmt_stage(new_stage)
        await callback.answer(f"Stage updated to {stage_label} ✅", show_alert=False)
        await show_lead_detail(callback, lead_id)
    else:
        error_detail = result.get("detail") if result else "API error"
        await safe_edit(
            callback,
            ui.format_error(f"Could not update stage for lead #{lead_id}.", error_detail),
            get_retry_keyboard(f"led{lead_id}_stage", f"lvw{lead_id}")
        )


# ─────────────────────────────────────────────────────────────
# Edit Source — Apply
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("edsrc"))
async def handle_edit_source(callback: CallbackQuery, state: FSMContext):
    raw = callback.data[5:]
    parts = raw.split("_", 1)
    lead_id_str = parts[0]
    new_source = parts[1] if len(parts) > 1 else ""

    try:
        lead_id = int(lead_id_str)
    except ValueError:
        await callback.answer("Invalid lead", show_alert=True)
        return

    result = await update_lead_via_api(lead_id, {"source": new_source}, user_id=callback.from_user.id)
    if result:
        await callback.answer(f"Source updated to {ui.fmt_source(new_source)} ✅", show_alert=False)
        await show_lead_detail(callback, lead_id)
    else:
        await safe_edit(
            callback,
            ui.format_error(f"Could not update source for lead #{lead_id}."),
            get_retry_keyboard(f"led{lead_id}_src", f"lvw{lead_id}")
        )


# ─────────────────────────────────────────────────────────────
# Edit Domain — Apply
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("eddom"))
async def handle_edit_domain(callback: CallbackQuery, state: FSMContext):
    raw = callback.data[5:]
    parts = raw.split("_", 1)
    lead_id_str = parts[0]
    new_domain_raw = parts[1] if len(parts) > 1 else ""
    domain_value = None if new_domain_raw == "none" else new_domain_raw

    try:
        lead_id = int(lead_id_str)
    except ValueError:
        await callback.answer("Invalid lead", show_alert=True)
        return

    result = await update_lead_via_api(lead_id, {"business_domain": domain_value}, user_id=callback.from_user.id)
    if result:
        label = ui.fmt_domain(domain_value) if domain_value else "Removed"
        await callback.answer(f"Domain → {label} ✅", show_alert=False)
        await show_lead_detail(callback, lead_id)
    else:
        await safe_edit(
            callback,
            ui.format_error(f"Could not update domain for lead #{lead_id}."),
            get_retry_keyboard(f"led{lead_id}_dom", f"lvw{lead_id}")
        )


# ─────────────────────────────────────────────────────────────
# Confirm Delete — Apply
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("cfddel"))
async def handle_confirm_delete(callback: CallbackQuery, state: FSMContext):
    raw = callback.data[6:]   # "cfddel" is 6 chars
    parts = raw.split("_", 1)
    lead_id_str = parts[0]
    confirm = parts[1] if len(parts) > 1 else ""

    try:
        lead_id = int(lead_id_str)
    except ValueError:
        await callback.answer("Invalid lead", show_alert=True)
        return

    if confirm == "y":
        await callback.answer("Deleting...")
        success = await delete_lead_via_api(lead_id, user_id=callback.from_user.id)
        if success:
            await safe_edit(
                callback,
                ui.format_success(f"Lead #{lead_id} has been permanently deleted."),
                get_leads_category_keyboard()
            )
        else:
            await safe_edit(
                callback,
                ui.format_error(f"Failed to delete lead #{lead_id}.", "The lead may have already been deleted."),
                get_back_keyboard(f"lvw{lead_id}")
            )
    else:
        await show_lead_detail(callback, lead_id)


# ─────────────────────────────────────────────────────────────
# Quick Stage Action Buttons (Contacted / Qualified / Transfer / Lost)
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("lac"))
async def handle_lead_action(callback: CallbackQuery, state: FSMContext):
    raw = callback.data[3:]
    parts = raw.split("_", 1)
    lead_id_str = parts[0]
    action = parts[1] if len(parts) > 1 else ""

    try:
        lead_id = int(lead_id_str)
    except ValueError:
        await callback.answer("Invalid lead", show_alert=True)
        return

    stage_map = {"c": "CONTACTED", "q": "QUALIFIED", "t": "TRANSFERRED", "l": "LOST"}
    new_stage = stage_map.get(action)

    if new_stage:
        await callback.answer(f"Marking as {ui.fmt_stage(new_stage)}...")
        result = await update_lead_stage_via_api(lead_id, new_stage, user_id=callback.from_user.id)
        if result and "error" not in result:
            await callback.answer(f"✅ Lead #{lead_id} → {ui.fmt_stage(new_stage)}", show_alert=False)
            await show_lead_detail(callback, lead_id)
        else:
            error_detail = result.get("detail") if result else "API error"
            await safe_edit(
                callback,
                ui.format_error(
                    f"Cannot move lead #{lead_id} to <b>{new_stage}</b>.",
                    error_detail
                ),
                get_retry_keyboard(f"lac{lead_id}_{action}", f"lvw{lead_id}")
            )
    elif action == "a":
        # AI Analyze
        await callback.answer("Running AI analysis...")
        result = await _api_post(f"/api/v1/leads/{lead_id}/analyze", {}, user_id=callback.from_user.id)
        if result and "error" not in result:
            score = result.get("score", 0)
            recommendation = result.get("recommendation", "N/A")
            reason = result.get("reason", "")
            text = (
                f"🤖 <b>AI ANALYSIS</b>  —  Lead #{lead_id}\n\n"
                f"<b>Score:</b>\n{ui.ai_score_bar(score)}\n\n"
                f"<b>💡 Recommendation:</b>  <i>{recommendation}</i>\n"
            )
            if reason:
                text += f"\n<b>📋 Reason:</b>\n<i>{reason}</i>"
            await safe_edit(callback, text, get_lead_detail_keyboard(lead_id))
        else:
            await safe_edit(
                callback,
                ui.format_error("AI analysis failed.", "The AI service may be unavailable."),
                get_back_keyboard(f"lvw{lead_id}")
            )


# ─────────────────────────────────────────────────────────────
# Add Note FSM
# ─────────────────────────────────────────────────────────────

@router.message(AddNoteState.waiting_for_text)
async def handle_note_text(message: Message, state: FSMContext):
    text = message.text or ""
    if not text:
        return
        
    if len(text) > 500:
        await message.answer(
            f"⚠️ Note is too long ({len(text)} chars). Max 500 characters.\n\n"
            f"Please shorten your note.",
            parse_mode="HTML"
        )
        return

    data = await state.get_data()
    lead_id = data.get("note_lead_id")
    await state.update_data(pending_note_text=text)
    await state.set_state(AddNoteState.waiting_for_confirm)
    
    await message.answer(
        ui.format_note_confirm(lead_id, text),
        reply_markup=get_note_confirm_keyboard(lead_id),
        parse_mode="HTML"
    )


@router.callback_query(F.data.startswith("ntcf"), AddNoteState.waiting_for_confirm)
async def handle_note_confirm(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split("_")
    action = parts[1] # s, e, d
    lead_id = parts[0].replace("ntcf", "")
    
    data = await state.get_data()
    text = data.get("pending_note_text")
    
    if action == "s":
        await callback.answer("Saving...")
        result = await _api_post(f"/api/v1/leads/{lead_id}/notes", {"content": text}, user_id=callback.from_user.id)
        await state.clear()
        if result and "error" not in result:
            await safe_edit(callback, ui.format_success(f"Note saved to Lead #{lead_id}."), get_back_keyboard(f"lvw{lead_id}"))
        else:
            error_detail = result.get("detail") if result else "API error"
            await safe_edit(callback, ui.format_error(f"Failed to save note.", error_detail), get_back_keyboard(f"led{lead_id}_ntm"))
            
    elif action == "e":
        await state.set_state(AddNoteState.waiting_for_text)
        await safe_edit(callback, ui.format_note_prompt(lead_id), get_note_cancel_keyboard(lead_id))
        
    elif action == "d":
        await state.clear()
        await safe_edit(callback, ui.format_notes_menu(lead_id, 0), get_notes_manage_keyboard(lead_id))


@router.callback_query(F.data.startswith("ntp"))
async def handle_note_pagination(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split("_")
    lead_id = parts[0].replace("ntp", "")
    idx = int(parts[1])
    
    data = await state.get_data()
    notes = data.get("viewing_notes", [])
    
    if not notes or idx >= len(notes):
        await callback.answer("No more notes.")
        return
        
    note = notes[idx]
    await safe_edit(
        callback,
        ui.format_single_note(lead_id, note, idx, len(notes)),
        get_note_view_keyboard(lead_id, note['id'], idx, len(notes))
    )


@router.callback_query(F.data.startswith("ntdel"))
async def handle_note_delete(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split("_")
    lead_id = parts[0].replace("ntdel", "")
    note_id = parts[1]
    
    await callback.answer("Deleting...")
    from app.bot.handlers import _api_delete
    success = await _api_delete(f"/api/v1/leads/{lead_id}/notes/{note_id}", user_id=callback.from_user.id)
    
    if success:
        # Refresh the notes list in state if possible, or just go back to menu
        await callback.answer("Note deleted successfully.")
        notes_data = await _api_get(f"/api/v1/leads/{lead_id}/notes", user_id=callback.from_user.id)
        count = notes_data.get("total", 0) if notes_data else 0
        await safe_edit(
            callback,
            ui.format_notes_menu(lead_id, count),
            get_notes_manage_keyboard(lead_id, has_notes=(count > 0))
        )
    else:
        await callback.answer("Failed to delete note.", show_alert=True)


# ─────────────────────────────────────────────────────────────
# Search FSM
# ─────────────────────────────────────────────────────────────

@router.message(SearchState.waiting_for_query)
async def handle_search_query(message: Message, state: FSMContext):
    query_text = (message.text or "").strip()
    if not query_text:
        await message.answer("⚠️ Please enter a search query.", parse_mode="HTML")
        return

    await state.clear()
    await message.answer(f"🔍 <i>Searching for <b>{query_text}</b>...</i>", parse_mode="HTML")

    # Parse complex query tokens
    search_params = {"user_id": message.from_user.id}
    clean_query = []
    
    tokens = query_text.split()
    for token in tokens:
        if ":" in token:
            try:
                key, val = token.split(":", 1)
                key = key.lower()
                if key == "domain":
                    search_params["domain"] = val
                elif key == "stage":
                    search_params["stage"] = val
                elif key == "source":
                    search_params["source"] = val
            except ValueError:
                clean_query.append(token)
        else:
            clean_query.append(token)
            
    if clean_query:
        search_params["query"] = " ".join(clean_query)
        
    results = await get_leads_via_api(**search_params)

    if results:
        header = ui.format_leads_list(results, f"🔍 Results: \"{query_text}\"", 0, 1)
        keyboard = get_lead_list_keyboard(results[:LEADS_PAGE_SIZE], 0, 1)
        await message.answer(header, reply_markup=keyboard, parse_mode="HTML")
    else:
        await message.answer(
            f"🔍 <b>No results</b> for <i>\"{query}\"</i>\n\n"
            f"Try searching by Lead ID, domain, source, or stage.",
            reply_markup=get_back_to_menu_keyboard(),
            parse_mode="HTML"
        )


# ─────────────────────────────────────────────────────────────
# Quick Actions
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data == "quick_myleads")
async def quick_myleads(callback: CallbackQuery, state: FSMContext):
    await callback.answer("Loading your leads...")
    user_id = callback.from_user.id
    all_leads = await get_leads_via_api(user_id=user_id)
    my_leads = [l for l in all_leads if str(l.get("telegram_id")) == str(user_id)]
    await show_leads_list_page(callback, my_leads, "👤 My Leads")


@router.callback_query(F.data == "quick_refresh")
async def quick_refresh(callback: CallbackQuery, state: FSMContext):
    await callback.answer("Refreshed! ✅")
    stats = await get_dashboard_via_api(user_id=callback.from_user.id)
    if stats and "error" not in stats:
        await safe_edit(callback, ui.format_dashboard(stats), get_dashboard_keyboard())
    else:
        leads = await get_leads_via_api(user_id=callback.from_user.id)
        await safe_edit(callback, ui.format_stats_simple(leads), get_dashboard_keyboard())


@router.callback_query(F.data == "goto_advanced_stats")
async def handle_advanced_stats(callback: CallbackQuery, state: FSMContext):
    await callback.answer("Analyzing deep data... 📊")
    # Fetch from the new advanced API endpoint
    data = await _api_get("/api/v1/dashboard/advanced", user_id=callback.from_user.id)
    
    if data and "error" not in data:
        await safe_edit(
            callback, 
            ui.format_advanced_stats(data), 
            get_back_keyboard("goto_dashboard")
        )
    else:
        await callback.answer("Failed to load advanced analytics.", show_alert=True)


@router.callback_query(F.data == "export_csv")
async def handle_export_csv(callback: CallbackQuery, state: FSMContext):
    await callback.answer("Triggering export...")
    user_id = callback.from_user.id
    # Call export API
    result = await _api_post(f"/api/v1/export?admin_id={user_id}", {}, user_id=user_id)
    
    if result and "error" not in result:
        await callback.answer("Export started! 📊\nYou will receive the CSV file shortly.", show_alert=True)
    else:
        await callback.answer("Failed to start export. Admin rights required.", show_alert=True)



# ─────────────────────────────────────────────────────────────
# New Lead — FSM Source → Domain → Create
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Advanced Lead Capture FSM
# ─────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("src_"))
async def handle_source(callback: CallbackQuery, state: FSMContext):
    source = callback.data.replace("src_", "")
    await state.update_data(source=source)
    await state.set_state(LeadCreationState.waiting_for_name)
    await safe_edit(
        callback,
        ui.format_lead_creation_step("1/8", "👤 ПОВНЕ ІМ'Я", "Будь ласка, введіть повне ім'я ліда або назву компанії:", hint="Це основний ідентифікатор у системі."),
        get_name_keyboard()
    )

@router.message(LeadCreationState.waiting_for_name)
async def handle_name(message: Message, state: FSMContext):
    name = (message.text or "").strip()
    if len(name) < 2:
        await message.answer("⚠️ Ім'я занадто коротке. Будь ласка, введіть коректне ім'я.")
        return
    await state.update_data(full_name=name)
    await state.set_state(LeadCreationState.waiting_for_email)
    await message.answer(
        ui.format_lead_creation_step("2/8", "📧 EMAIL", "Введіть діючу email-адресу клієнта:", hint="Email використовується для розсилок та автоматизації маркетингу."),
        reply_markup=get_email_keyboard(),
        parse_mode="HTML"
    )

@router.message(LeadCreationState.waiting_for_email)
async def handle_email(message: Message, state: FSMContext):
    email = (message.text or "").strip()
    if not is_valid_email(email):
        await message.answer("⚠️ Invalid email format. Please try again or skip.")
        return
    await state.update_data(email=email)
    await proceed_to_phone(message, state)

@router.callback_query(F.data == "skip_email", LeadCreationState.waiting_for_email)
async def skip_email(callback: CallbackQuery, state: FSMContext):
    await state.update_data(email=None)
    await proceed_to_phone(callback.message, state)

async def proceed_to_phone(msg: Message, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_phone)
    await msg.answer(
        ui.format_lead_creation_step("3/8", "📞 PHONE NUMBER", "Enter phone number (e.g. +1234567890):"),
        reply_markup=get_phone_keyboard(),
        parse_mode="HTML"
    )

@router.message(LeadCreationState.waiting_for_phone)
async def handle_phone(message: Message, state: FSMContext):
    phone = (message.text or "").strip()
    if not is_valid_phone(phone):
        await message.answer("⚠️ Invalid phone format. Please try again or skip.")
        return
    await state.update_data(phone=phone)
    await proceed_to_username(message, state)

@router.callback_query(F.data == "skip_phone", LeadCreationState.waiting_for_phone)
async def skip_phone(callback: CallbackQuery, state: FSMContext):
    await state.update_data(phone=None)
    await proceed_to_username(callback.message, state)

async def proceed_to_username(msg: Message, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_username)
    await msg.answer(
        ui.format_lead_creation_step("4/8", "📡 MESSENGER USERNAME", "Enter Telegram/WhatsApp username:"),
        reply_markup=get_username_keyboard(),
        parse_mode="HTML"
    )

@router.message(LeadCreationState.waiting_for_username)
async def handle_username_msg(message: Message, state: FSMContext):
    await state.update_data(external_username=message.text)
    await proceed_to_domain(message, state)

@router.callback_query(F.data == "skip_username", LeadCreationState.waiting_for_username)
async def skip_username(callback: CallbackQuery, state: FSMContext):
    await state.update_data(external_username=None)
    await proceed_to_domain(callback.message, state)

async def proceed_to_domain(msg: Message, state: FSMContext):
    await state.set_state(LeadCreationState.waiting_for_domain)
    await msg.answer(
        ui.format_lead_creation_step("5/8", "🌐СФЕРА БІЗНЕСУ", "Оберіть галузь, у якій працює лід:", hint="Від цього залежить стратегія спілкування та релевантність кейсів."),
        reply_markup=get_domain_keyboard("5/8"),
        parse_mode="HTML"
    )

@router.callback_query(F.data.startswith("dmn_"), LeadCreationState.waiting_for_domain)
async def handle_domain_step(callback: CallbackQuery, state: FSMContext):
    domain_raw = callback.data.replace("dmn_", "")
    domain = None if domain_raw == "skip" else domain_raw
    await state.update_data(business_domain=domain)
    await state.set_state(LeadCreationState.waiting_for_intent)
    await safe_edit(
        callback,
        ui.format_lead_creation_step("6/8", "🎯 ІНТЕНТ / ДІЯ", "Яку дію вчинив лід або який його запит?", hint="Наприклад: 'Зареєструвався' або 'Замовив дзвінок'."),
        get_intent_keyboard()
    )

@router.callback_query(F.data.startswith("int_"), LeadCreationState.waiting_for_intent)
async def handle_intent_step(callback: CallbackQuery, state: FSMContext):
    intent_map = {
        "reg": "Registration",
        "call": "Callback Request",
        "magnet": "Lead Magnet Download",
        "msg": "Direct Message",
        "skip": None
    }
    key = callback.data.replace("int_", "")
    await state.update_data(intent=intent_map.get(key))
    
    await state.set_state(LeadCreationState.waiting_for_qualification)
    await state.update_data(qual_step="company")
    await safe_edit(
        callback,
        ui.format_lead_creation_step("7/8", "🏢 COMPANY NAME", "Enter company name (B2B qualification):"),
        get_qualification_keyboard("Company")
    )

@router.message(LeadCreationState.waiting_for_qualification)
async def handle_qualification_msg(message: Message, state: FSMContext):
    data = await state.get_data()
    qstep = data.get("qual_step")
    text = message.text
    
    if qstep == "company":
        await state.update_data(company=text, qual_step="position")
        await message.answer(
            ui.format_lead_creation_step("7/8", "👔 ROLE/POSITION", f"Company: <b>{text}</b>\n\nWhat is their role?"),
            reply_markup=get_qualification_keyboard("Role"),
            parse_mode="HTML"
        )
    elif qstep == "position":
        await state.update_data(position=text, qual_step="budget")
        await message.answer(
            ui.format_lead_creation_step("7/8", "💰 BUDGET", f"What is their target budget?"),
            reply_markup=get_qualification_keyboard("Budget"),
            parse_mode="HTML"
        )
    elif qstep == "budget":
        await state.update_data(budget=text, qual_step="pain")
        await message.answer(
            ui.format_lead_creation_step("7/8", "🩹 PAIN POINTS", "What problem are they trying to solve?"),
            reply_markup=get_qualification_keyboard("Pain"),
            parse_mode="HTML"
        )
    elif qstep == "pain":
        await state.update_data(pain_points=text)
        await proceed_to_confirm_creation(message, state)

@router.callback_query(F.data == "qual_skip", LeadCreationState.waiting_for_qualification)
async def skip_qual_step(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    qstep = data.get("qual_step")
    
    if qstep == "company":
        await state.update_data(company=None, qual_step="position")
        await safe_edit(callback, ui.format_lead_creation_step("7/8", "👔 ROLE/POSITION", "What is their role?"), get_qualification_keyboard("Role"))
    elif qstep == "position":
        await state.update_data(position=None, qual_step="budget")
        await safe_edit(callback, ui.format_lead_creation_step("7/8", "💰 BUDGET", "What is their target budget?"), get_qualification_keyboard("Budget"))
    elif qstep == "budget":
        await state.update_data(budget=None, qual_step="pain")
        await safe_edit(callback, ui.format_lead_creation_step("7/8", "🩹 PAIN POINTS", "What problem are they trying to solve?"), get_qualification_keyboard("Pain"))
    elif qstep == "pain":
        await state.update_data(pain_points=None)
        await proceed_to_confirm_creation(callback.message, state)

@router.callback_query(F.data == "qual_abort", LeadCreationState.waiting_for_qualification)
async def abort_qualification(callback: CallbackQuery, state: FSMContext):
    await state.update_data(company=None, position=None, budget=None, pain_points=None)
    await proceed_to_confirm_creation(callback.message, state)

async def proceed_to_confirm_creation(msg: Message, state: FSMContext):
    data = await state.get_data()
    await state.set_state(LeadCreationState.confirm)
    await msg.answer(
        ui.format_lead_confirm_card(data),
        reply_markup=get_lead_confirm_keyboard(),
        parse_mode="HTML"
    )

@router.callback_query(F.data == "cf_create", LeadCreationState.confirm)
async def finalize_lead_creation(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    await callback.answer("Creating lead...")
    # Inject telegram_id
    data["telegram_id"] = str(callback.from_user.id)
    
    lead = await _api_post("/api/v1/leads", data, user_id=callback.from_user.id)
    await state.clear()

    if lead:
        text = (
            f"✅ <b>Lead Created!</b>\n\n"
            f"<b>ID:</b>  #{lead['id']}\n"
            f"<b>Name:</b> {lead.get('full_name')}\n"
            f"<b>Stage:</b> {ui.fmt_stage(lead.get('stage'))}\n\n"
            f"<i>Tap below to view or manage this lead.</i>"
        )
        builder = InlineKeyboardBuilder()
        builder.add(InlineKeyboardButton(text=f"📄 Open Lead #{lead['id']}", callback_data=f"lvw{lead['id']}"))
        builder.add(InlineKeyboardButton(text="🏠 Main Menu", callback_data="goto_menu"))
        builder.adjust(1)
        await safe_edit(callback, text, builder.as_markup())
    else:
        await safe_edit(callback, ui.format_error("Failed to create lead."), get_retry_keyboard("goto_newlead", "goto_menu"))


# ─────────────────────────────────────────────────────────────
# Error Handler
# ─────────────────────────────────────────────────────────────

@router.errors()
async def handle_errors(event: Exception):
    logger.error(f"Bot error: {event}")


# ─────────────────────────────────────────────────────────────
# Dispatcher Setup
# ─────────────────────────────────────────────────────────────

def get_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.include_router(router)
    return dp


async def start_bot():
    bot_instance = get_bot()
    dp = get_dispatcher()
    await dp.start_polling(bot_instance)


if __name__ == "__main__":
    import asyncio
    asyncio.run(start_bot())
