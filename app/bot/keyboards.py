"""
Telegram Bot keyboard layouts — Professional UI/UX with emoji and visual hierarchy.
All keyboards use consistent emoji language defined in ui.py.
"""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder

from app.bot.ui import STAGE_META, SOURCE_META, DOMAIN_META


# ─────────────────────────────────────────────────────────────
# Reply Keyboard (DISABLED - removed main menu per user request)
# ─────────────────────────────────────────────────────────────

def get_main_menu_keyboard() -> None:
    """Main menu keyboard disabled."""
    return None


# ─────────────────────────────────────────────────────────────
# Start / Main Menu (Inline - DISABLED)
# ─────────────────────────────────────────────────────────────

def get_start_keyboard() -> InlineKeyboardMarkup:
    """Welcome screen inline menu."""
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="📊 Dashboard",  callback_data="goto_dashboard"))
    builder.add(InlineKeyboardButton(text="📋 Leads",      callback_data="goto_leads"))
    builder.add(InlineKeyboardButton(text="💰 Sales",      callback_data="goto_sales"))
    builder.add(InlineKeyboardButton(text="➕ New Lead",   callback_data="goto_newlead"))
    builder.add(InlineKeyboardButton(text="📋 Paste Lead",callback_data="goto_paste_lead"))
    builder.add(InlineKeyboardButton(text="🔍 Search",     callback_data="goto_search"))
    builder.add(InlineKeyboardButton(text="⚙️ Settings",   callback_data="goto_settings"))
    builder.adjust(2, 2, 3)
    return builder.as_markup()


def get_menu_keyboard() -> InlineKeyboardMarkup:
    """Standard inline main menu."""
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="📊 Dashboard",  callback_data="goto_dashboard"))
    builder.add(InlineKeyboardButton(text="📋 Leads",      callback_data="goto_leads"))
    builder.add(InlineKeyboardButton(text="➕ New Lead",   callback_data="goto_newlead"))
    builder.add(InlineKeyboardButton(text="📋 Paste Lead",callback_data="goto_paste_lead"))
    builder.add(InlineKeyboardButton(text="🔍 Search",     callback_data="goto_search"))
    builder.add(InlineKeyboardButton(text="⚡ Quick",      callback_data="goto_quick"))
    builder.add(InlineKeyboardButton(text="⚙️ Settings",   callback_data="goto_settings"))
    builder.adjust(2, 2, 3)
    return builder.as_markup()


def get_paste_lead_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for paste lead flow."""
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="📋 Paste Lead Data", callback_data="start_paste_lead"))
    builder.add(InlineKeyboardButton(text="❌ Cancel", callback_data="goto_menu"))
    builder.adjust(1, 1)
    return builder.as_markup()


def get_paste_confirm_keyboard() -> InlineKeyboardMarkup:
    """Confirmation keyboard after parsing pasted data."""
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="✅ Create Lead", callback_data="paste_create"))
    builder.add(InlineKeyboardButton(text="✏️ Edit", callback_data="paste_edit"))
    builder.add(InlineKeyboardButton(text="❌ Cancel", callback_data="goto_menu"))
    builder.adjust(1, 1, 1)
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Leads Categories
# ─────────────────────────────────────────────────────────────

def get_leads_category_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="👤 My Leads",   callback_data="filter_myleads"))
    builder.add(InlineKeyboardButton(text="📈 By Stage",   callback_data="cat_stage"))
    builder.add(InlineKeyboardButton(text="📥 By Source",  callback_data="cat_source"))
    builder.add(InlineKeyboardButton(text="🏢 By Domain",  callback_data="cat_domain"))
    builder.add(InlineKeyboardButton(text="📋 All Leads",  callback_data="filter_all"))
    builder.add(InlineKeyboardButton(text="🏠 Menu",       callback_data="goto_menu"))
    builder.adjust(1, 2, 2, 1)
    return builder.as_markup()


def get_stage_subcategories_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for stage, meta in STAGE_META.items():
        builder.add(InlineKeyboardButton(
            text=f"{meta['emoji']} {meta['label']}",
            callback_data=f"filter_{stage.lower()}"
        ))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data="goto_leads"))
    builder.adjust(1, 1, 1, 1, 1, 1)
    return builder.as_markup()


def get_source_subcategories_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for source, meta in SOURCE_META.items():
        builder.add(InlineKeyboardButton(
            text=f"{meta['emoji']} {meta['label']}",
            callback_data=f"filter_{source.lower()}"
        ))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data="goto_leads"))
    builder.adjust(1, 1, 1, 1)
    return builder.as_markup()


def get_domain_subcategories_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for domain, meta in DOMAIN_META.items():
        builder.add(InlineKeyboardButton(
            text=f"{meta['emoji']} {meta['label']}",
            callback_data=f"filter_{domain.lower()}"
        ))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data="goto_leads"))
    builder.adjust(1, 1, 1, 1)
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Lead List Keyboard (with per-row lead entries)
# ─────────────────────────────────────────────────────────────

def get_lead_list_keyboard(
    leads: list,
    page: int = 0,
    total_pages: int = 1,
    back_cb: str = "goto_leads"
) -> InlineKeyboardMarkup:
    """Show each lead as a button row. Paginated."""
    builder = InlineKeyboardBuilder()

    for lead in leads:
        lead_id = lead.get("id", "?")
        stage = lead.get("stage", "new")
        domain = lead.get("business_domain")
        source = lead.get("source", "")

        stage_emoji = STAGE_META.get(stage, {}).get("emoji", "❓")
        domain_str = DOMAIN_META.get(domain, {}).get("label", "—") if domain else "—"
        src_emoji = SOURCE_META.get(source, {}).get("emoji", "•")

        ai_sc = lead.get("ai_score")
        score_str = f"  🤖{round(ai_sc * 100)}%" if ai_sc is not None else ""

        label = f"#{lead_id}  {stage_emoji} {domain_str} {src_emoji}{score_str}"
        builder.add(InlineKeyboardButton(text=label, callback_data=f"lvw{lead_id}"))

    # Pagination row
    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton(text="‹ Prev", callback_data=f"pg{page - 1}"))
        nav.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton(text="Next ›", callback_data=f"pg{page + 1}"))
        builder.row(*nav)

    builder.row(InlineKeyboardButton(text="‹ Back to Filters", callback_data=back_cb))
    builder.adjust(1)
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Lead Detail Keyboard
# ─────────────────────────────────────────────────────────────

def get_lead_detail_keyboard(lead_id, current_stage: str = None) -> InlineKeyboardMarkup:
    """Full action keyboard for lead detail view."""
    lid = str(lead_id)
    builder = InlineKeyboardBuilder()

    # Row 1: Edit fields
    builder.row(
        InlineKeyboardButton(text="✏️ Stage",  callback_data=f"led{lid}_stage"),
        InlineKeyboardButton(text="✏️ Source", callback_data=f"led{lid}_src"),
        InlineKeyboardButton(text="✏️ Domain", callback_data=f"led{lid}_dom"),
    )

    # Row 2: Quick stage actions
    builder.row(
        InlineKeyboardButton(text="📞 Contact",  callback_data=f"lac{lid}_c"),
        InlineKeyboardButton(text="✅ Qualify",  callback_data=f"lac{lid}_q"),
        InlineKeyboardButton(text="🚀 Transfer", callback_data=f"lac{lid}_t"),
        InlineKeyboardButton(text="❌ Lost",     callback_data=f"lac{lid}_l"),
    )

    # Row 3: Tools
    builder.row(
        InlineKeyboardButton(text="🤖 AI Analyze", callback_data=f"lac{lid}_a"),
        InlineKeyboardButton(text="📝 Notes",      callback_data=f"led{lid}_ntm"),
    )

    # Row 4: Danger + Back
    builder.row(
        InlineKeyboardButton(text="🗑 Delete",         callback_data=f"led{lid}_del"),
        InlineKeyboardButton(text="‹ Back to List",    callback_data="goto_leads"),
    )

    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Edit Stage Keyboard (highlights current)
# ─────────────────────────────────────────────────────────────

def get_edit_stage_keyboard(lead_id, current_stage: str = None) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    lid = str(lead_id)
    for stage, meta in STAGE_META.items():
        is_current = stage == current_stage
        label = f"{'✅ ' if is_current else ''}{meta['emoji']} {meta['label']}"
        builder.add(InlineKeyboardButton(
            text=label,
            callback_data=f"eds{lid}_{stage}"
        ))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data=f"lvw{lid}"))
    builder.adjust(1, 1, 1, 1, 1, 1)
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Edit Source Keyboard
# ─────────────────────────────────────────────────────────────

def get_edit_source_keyboard(lead_id, current_source: str = None) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    lid = str(lead_id)
    for source, meta in SOURCE_META.items():
        is_current = source == current_source
        label = f"{'✅ ' if is_current else ''}{meta['emoji']} {meta['label']}"
        builder.add(InlineKeyboardButton(
            text=label,
            callback_data=f"edsrc{lid}_{source}"
        ))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data=f"lvw{lid}"))
    builder.adjust(1, 1, 1, 1)
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Edit Domain Keyboard
# ─────────────────────────────────────────────────────────────

def get_edit_domain_keyboard(lead_id, current_domain: str = None) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    lid = str(lead_id)
    for domain, meta in DOMAIN_META.items():
        is_current = domain == current_domain
        label = f"{'✅ ' if is_current else ''}{meta['emoji']} {meta['label']}"
        builder.add(InlineKeyboardButton(
            text=label,
            callback_data=f"eddom{lid}_{domain}"
        ))
    builder.add(InlineKeyboardButton(
        text="🚫 Remove Domain",
        callback_data=f"eddom{lid}_none"
    ))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data=f"lvw{lid}"))
    builder.adjust(1, 1, 1, 1, 1)
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Confirm Delete
# ─────────────────────────────────────────────────────────────

def get_confirm_delete_keyboard(lead_id) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    lid = str(lead_id)
    builder.row(
        InlineKeyboardButton(text="⚠️ Yes, Delete Forever", callback_data=f"cfddel{lid}_y"),
    )
    builder.row(
        InlineKeyboardButton(text="✖ Cancel",  callback_data=f"lvw{lid}"),
    )
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# New Lead FSM Keyboards
# ─────────────────────────────────────────────────────────────

def get_source_keyboard(step: str = "1/3") -> InlineKeyboardMarkup:
    """Step 1: Choose source."""
    builder = InlineKeyboardBuilder()
    for source, meta in SOURCE_META.items():
        builder.add(InlineKeyboardButton(
            text=f"{meta['emoji']} {meta['label']}",
            callback_data=f"src_{source}"
        ))
    builder.add(InlineKeyboardButton(text="‹ Cancel", callback_data="goto_menu"))
    builder.adjust(3, 1)
    return builder.as_markup()


def get_name_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data="goto_newlead"))
    return builder.as_markup()


def get_email_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="⏭ Skip", callback_data="skip_email"))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data="back_name"))
    builder.adjust(1, 1)
    return builder.as_markup()


def get_phone_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="⏭ Skip", callback_data="skip_phone"))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data="back_email"))
    builder.adjust(1, 1)
    return builder.as_markup()


def get_username_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="⏭ Skip", callback_data="skip_username"))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data="back_phone"))
    builder.adjust(1, 1)
    return builder.as_markup()


def get_domain_keyboard(step: str = "2/3") -> InlineKeyboardMarkup:
    """Step 2: Choose domain."""
    builder = InlineKeyboardBuilder()
    for domain, meta in DOMAIN_META.items():
        builder.add(InlineKeyboardButton(
            text=f"{meta['emoji']} {meta['label']}",
            callback_data=f"dmn_{domain}"
        ))
    builder.add(InlineKeyboardButton(text="⏭ Skip", callback_data="dmn_skip"))
    builder.add(InlineKeyboardButton(text="‹ Back",  callback_data="back_username"))
    builder.adjust(3, 1, 1)
    return builder.as_markup()


def get_intent_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    texts = [
        ("📝 Registration", "int_reg"),
        ("📞 Callback",     "int_call"),
        ("📥 Lead Magnet",   "int_magnet"),
        ("💬 Message",       "int_msg"),
    ]
    for txt, cb in texts:
        builder.add(InlineKeyboardButton(text=txt, callback_data=cb))
    builder.add(InlineKeyboardButton(text="⏭ Skip", callback_data="int_skip"))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data="back_domain"))
    builder.adjust(2, 2, 1, 1)
    return builder.as_markup()


def get_qualification_keyboard(step: str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="⏭ Skip Step", callback_data="qual_skip"))
    builder.add(InlineKeyboardButton(text="❌ Skip Qualification", callback_data="qual_abort"))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data="back_intent"))
    builder.adjust(1, 1, 1)
    return builder.as_markup()


def get_lead_confirm_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="✅ Create Lead", callback_data="cf_create"))
    builder.add(InlineKeyboardButton(text="✏️ Edit", callback_data="back_intent"))
    builder.add(InlineKeyboardButton(text="✖ Discard", callback_data="goto_menu"))
    builder.adjust(1, 1, 1)
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Note Add
# ─────────────────────────────────────────────────────────────

def get_notes_manage_keyboard(lead_id, has_notes: bool = True) -> InlineKeyboardMarkup:
    """Notes management menu (Add / View)."""
    builder = InlineKeyboardBuilder()
    lid = str(lead_id)
    builder.row(InlineKeyboardButton(text="➕ Add New Note", callback_data=f"led{lid}_ntadd"))
    if has_notes:
        builder.row(InlineKeyboardButton(text="👁 View All Notes", callback_data=f"led{lid}_ntvw"))
    builder.row(InlineKeyboardButton(text="‹ Back", callback_data=f"lvw{lid}"))
    return builder.as_markup()


def get_note_view_keyboard(lead_id, note_id, index: int, total: int) -> InlineKeyboardMarkup:
    """Pagination for viewing notes."""
    builder = InlineKeyboardBuilder()
    lid = str(lead_id)
    nid = str(note_id)
    
    # Nav row
    nav = []
    if total > 1:
        prev_idx = (index - 1) if index > 0 else total - 1
        next_idx = (index + 1) if index < total - 1 else 0
        nav.append(InlineKeyboardButton(text="‹ Prev", callback_data=f"ntp{lid}_{prev_idx}"))
        nav.append(InlineKeyboardButton(text=f"{index + 1}/{total}", callback_data="noop"))
        nav.append(InlineKeyboardButton(text="Next ›", callback_data=f"ntp{lid}_{next_idx}"))
        builder.row(*nav)

    builder.row(
        InlineKeyboardButton(text="🗑 Delete Note", callback_data=f"ntdel{lid}_{nid}"),
        InlineKeyboardButton(text="‹ Back",        callback_data=f"led{lid}_ntm"),
    )
    return builder.as_markup()


def get_note_confirm_keyboard(lead_id) -> InlineKeyboardMarkup:
    """Confirmation before saving a note."""
    builder = InlineKeyboardBuilder()
    lid = str(lead_id)
    builder.row(
        InlineKeyboardButton(text="✅ Save",    callback_data=f"ntcf{lid}_s"),
        InlineKeyboardButton(text="✏️ Edit",    callback_data=f"ntcf{lid}_e"),
        InlineKeyboardButton(text="✖ Discard", callback_data=f"ntcf{lid}_d"),
    )
    return builder.as_markup()


def get_note_cancel_keyboard(lead_id, back_to_menu: bool = True) -> InlineKeyboardMarkup:
    """Cancel button during note-adding FSM."""
    builder = InlineKeyboardBuilder()
    cb = f"led{lead_id}_ntm" if back_to_menu else f"lvw{lead_id}"
    builder.add(InlineKeyboardButton(text="✖ Cancel", callback_data=cb))
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────

def get_search_cancel_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="✖ Cancel", callback_data="goto_menu"))
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Quick Actions
# ─────────────────────────────────────────────────────────────

def get_quick_actions_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="📊 Dashboard",   callback_data="goto_dashboard"))
    builder.add(InlineKeyboardButton(text="👤 My Leads",    callback_data="quick_myleads"))
    builder.add(InlineKeyboardButton(text="➕ Add Lead",    callback_data="goto_newlead"))
    builder.add(InlineKeyboardButton(text="🔄 Refresh",     callback_data="quick_refresh"))
    builder.add(InlineKeyboardButton(text="🔍 Search",      callback_data="goto_search"))
    builder.add(InlineKeyboardButton(text="🏠 Main Menu",   callback_data="goto_menu"))
    builder.adjust(2, 2, 2)
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────

def get_settings_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="🔔 Notifications", callback_data="settings_notif"))
    builder.add(InlineKeyboardButton(text="🤖 AI Settings",   callback_data="settings_ai"))
    builder.add(InlineKeyboardButton(text="👤 My Profile",    callback_data="settings_profile"))
    builder.add(InlineKeyboardButton(text="🏠 Main Menu",     callback_data="goto_menu"))
    builder.adjust(2, 1, 1)
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────

def get_dashboard_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="📊 Advanced Report", callback_data="goto_advanced_stats"))
    builder.row(
        InlineKeyboardButton(text="💰 Sales",          callback_data="goto_sales"),
        InlineKeyboardButton(text="📊 Export CSV",    callback_data="export_csv"),
    )
    builder.row(
        InlineKeyboardButton(text="🔄 Refresh",    callback_data="goto_dashboard"),
        InlineKeyboardButton(text="🏠 Main Menu",  callback_data="goto_menu"),
    )
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Sales
# ─────────────────────────────────────────────────────────────

def get_sales_category_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="👤 My Sales", callback_data="filter_mysales"))
    builder.add(InlineKeyboardButton(text="📈 By Stage", callback_data="scat_stage"))
    builder.add(InlineKeyboardButton(text="💰 All Sales", callback_data="filter_sales_all"))
    builder.add(InlineKeyboardButton(text="🏠 Menu", callback_data="goto_menu"))
    builder.adjust(1, 2, 1)
    return builder.as_markup()


def get_sale_stage_categories_keyboard() -> InlineKeyboardMarkup:
    from app.bot.ui import SALE_STAGE_META
    builder = InlineKeyboardBuilder()
    for stage, meta in SALE_STAGE_META.items():
        builder.add(InlineKeyboardButton(
            text=f"{meta['emoji']} {meta['label']}",
            callback_data=f"sfilter_{stage}"
        ))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data="goto_sales"))
    builder.adjust(1)
    return builder.as_markup()


def get_sales_list_keyboard(
    sales: list,
    page: int = 0,
    total_pages: int = 1,
    back_cb: str = "goto_sales"
) -> InlineKeyboardMarkup:
    """Show each sale as a button row. Paginated."""
    builder = InlineKeyboardBuilder()
    from app.bot.ui import SALE_STAGE_META

    for sale in sales:
        sale_id = sale.get("id", "?")
        stage = sale.get("stage", "new")
        amount = sale.get("amount")
        
        stage_emoji = SALE_STAGE_META.get(stage, {}).get("emoji", "❓")
        amount_str = f"  💰{amount / 100:.0f}" if amount is not None else ""
        
        label = f"#{sale_id}  {stage_emoji}{amount_str}"
        builder.add(InlineKeyboardButton(text=label, callback_data=f"svw{sale_id}"))

    # Pagination row
    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton(text="‹ Prev", callback_data=f"spg{page - 1}"))
        nav.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="snoop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton(text="Next ›", callback_data=f"spg{page + 1}"))
        builder.row(*nav)

    builder.row(InlineKeyboardButton(text="‹ Back to Filters", callback_data=back_cb))
    builder.adjust(1)
    return builder.as_markup()


def get_sale_detail_keyboard(sale_id, current_stage: str = None) -> InlineKeyboardMarkup:
    """Full action keyboard for sale detail view."""
    sid = str(sale_id)
    builder = InlineKeyboardBuilder()

    builder.row(
        InlineKeyboardButton(text="✏️ Stage",  callback_data=f"sed{sid}_stage"),
        InlineKeyboardButton(text="💰 Amount", callback_data=f"sed{sid}_amt"),
    )
    builder.row(
        InlineKeyboardButton(text="📝 Notes",  callback_data=f"sed{sid}_nt"),
        InlineKeyboardButton(text="🗑 Delete", callback_data=f"sed{sid}_del"),
    )
    builder.row(
        InlineKeyboardButton(text="📄 Client Lead", callback_data=f"sed{sid}_lview"),
        InlineKeyboardButton(text="‹ Back to List", callback_data="goto_sales"),
    )

    return builder.as_markup()


def get_edit_sale_stage_keyboard(sale_id, current_stage: str = None) -> InlineKeyboardMarkup:
    from app.bot.ui import SALE_STAGE_META
    builder = InlineKeyboardBuilder()
    sid = str(sale_id)
    for stage, meta in SALE_STAGE_META.items():
        is_current = stage == current_stage
        label = f"{'✅ ' if is_current else ''}{meta['emoji']} {meta['label']}"
        builder.add(InlineKeyboardButton(
            text=label,
            callback_data=f"seds{sid}_{stage}"
        ))
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data=f"svw{sid}"))
    builder.adjust(1)
    return builder.as_markup()


# ─────────────────────────────────────────────────────────────
# Generic Helpers
# ─────────────────────────────────────────────────────────────

def get_back_keyboard(callback_data: str = "goto_menu") -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="‹ Back", callback_data=callback_data))
    return builder.as_markup()


def get_back_to_menu_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="🏠 Main Menu", callback_data="goto_menu"))
    return builder.as_markup()


def get_retry_keyboard(retry_cb: str, back_cb: str = "goto_menu") -> InlineKeyboardMarkup:
    """Retry + back buttons for error states."""
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="🔄 Try Again", callback_data=retry_cb))
    builder.add(InlineKeyboardButton(text="🏠 Menu",      callback_data=back_cb))
    builder.adjust(2)
    return builder.as_markup()
